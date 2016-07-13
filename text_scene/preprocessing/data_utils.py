import csv
import numpy as np
import pandas as pd
from math import ceil
from os.path import basename
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from paths import (
    CAPTIONS_FILE,
    SENTENCES_CSV,
    REJECTED_IMGS_FILE,
    ANNOTATED_IMGS_FILE,
    REDO_IMGS_FILE,
    IMG_URLS,
    COMBINED_MTURK_RESULTS_CSV,
    MTURK_RESULTS_CSV,
    BATCH_URLS_CSV,
    COMBINED_BATCH_RESULTS_CSV
)

q1map = {'0': 'indoors', '1': 'outdoors'}
q2map = {'0': 'man-made', '1': 'natural'}
q3map = {'0': 'transportation_urban',
         '1': 'restaurant',
         '2': 'recreation',
         '3': 'domestic',
         '4': 'work_education',
         '5': 'other_unclear',
         'NA': 'NA'}
q4map = {'0': 'body_of_water',
         '1': 'field_forest',
         '2': 'mountain',
         '3': 'other_unclear',
         'NA': 'NA'}

def make_datadict(results_csv, keep_url=False):
    """
    Read in the results of MTurk annotation and create
    a dictionary mapping images to question responses, where the
    responses are converted from integers to the corresponding label
    strings.

    Parameters
    ----------
    results_csv: path to csv file storing MTurk annotation results
    keep_url: boolean, if True, keep the image names (keys in the dict)
        otherwise, shorten the url to just the filename

    Returns
    -------
    datadict: dictionary mapping image keys to list of annotations, where
        an annotation is a list of 4 strings. The index of each string
        corresponds to which question in the MTurk template it's a
        response to.
    """
    datadict = defaultdict(list)
    with open(results_csv, 'r') as csvf:
        reader = csv.reader(csvf)
        next(reader)
        for row in reader:
            if keep_url:
                img_file = row[0]
            else:
                img_file = url2filename(row[0])
            q1 = q1map[row[2]] if row[2].isdigit() else row[2]
            q2 = q2map[row[3]] if row[3].isdigit() else row[3]
            q3 = q3map[row[4]] if (row[4].isdigit() or row[4] == 'NA') else row[4]
            q4 = q4map[row[5]] if (row[5].isdigit() or row[5] == 'NA') else row[5]
            datadict[img_file].append([q1, q2, q3, q4])
    return datadict

def write_sentence_csv(datadict, captions_file, out_csv):
    """
    Given a datadict constructed from a results csv, write a new
    csv file by mapping each annotated image to its 5 corresponding
    captions and labeling each caption with the label given in results csv.
    """
    with open(captions_file) as cfile, open(out_csv, 'wb') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['sentence', 'q1', 'q2', 'q3', 'q4', 'img_file'])
        n_sents = 0
        for line in cfile:
            split_line = line.split()
            split_line = [w if not w.isdigit()
                          else '<NUMBER>'
                          for w in split_line]
            img_file = split_line[0].split('#')[0]
            if img_file in datadict:
                n_sents += 1
                sentence = ' '.join(split_line[1:]).lower()
                annotations = datadict[img_file]
                if len(annotations) != 4: # need to have 4 annotation fields
                    annotations = annotations[0]
                assert len(annotations) == 4
                writer.writerow([sentence] + annotations + [img_file])
    print "Wrote sentence csv with %i sentences." % n_sents

def get_img_lists(img_url_file=IMG_URLS, log_file=ANNOTATED_IMGS_FILE,
                  keep_url=False):
    all_images = set()
    with open(img_url_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if keep_url:
                all_images.add(row[0])
            else:
                all_images.add(url2filename(row[0]))
    annotated = set()
    with open(log_file, 'r') as log:
        for line in log:
            if keep_url:
                annotated.add(line.strip())
            else:
                annotated.add(url2filename(line.strip()))
    not_annotated = list(all_images - annotated)
    return annotated, not_annotated

#########################################
### Annotation using MTurk layout/UI  ###
#########################################

def write_batch_urls_csv(img_urls, outcsv=BATCH_URLS_CSV, n_imgs=100):
    """get image urls for unannotated images and write them to
    `outcsv`."""
    with open(outcsv, 'wb') as out:
        writer = csv.writer(out)
        writer.writerow(['img_url'])
        for img_url in img_urls:
            writer.writerow([img_url.strip()])

def write_annotated_urls(img_urls, outfile=ANNOTATED_IMGS_FILE):
    """given a list of image urls, write them to `annotated_imgs.txt`."""
    with open(outfile, 'a') as f:
        for img_url in img_urls:
            f.write(img_url.strip() + '\n')

def make_batch(n_imgs=100):
    """Create image url csv file for images to be annotated in the batch
    and write those image urls to annotated_imgs.txt."""
    _, unannotated = get_img_lists(keep_url=True)
    to_annotate = np.random.choice(unannotated, size=(n_imgs,), replace=False)
    write_batch_urls_csv(to_annotate)
    write_annotated_urls(to_annotate)

def write_results_from_batch_csv(batch_csv, outcsv):
    """write results csv from mturk generated batch results.
    Writes to `outcsv`."""
    df = pd.read_csv(batch_csv)
    with open(outcsv, 'wb') as out:
        writer = csv.writer(out)
        writer.writerow(['image_url', 'worker_id', 'q1', 'q2', 'q3', 'q4'])
        for _, row in df.iterrows():
            img_url = row['Input.img_url'].strip()
            worker_id = row['WorkerId'].strip()
            q1 = int(row['Answer.Answer_1'])
            q2 = int(row['Answer.Answer_2'])
            try:
                q3 = int(row['Answer.Answer_3'])
            except ValueError:
                q3 = row['Answer.Answer_3']
            try:
                q4 = int(row['Answer.Answer_4'])
            except ValueError:
                q4 = row['Answer.Answer_4']
            if pd.isnull(q4):
                q4 = 'NA'
            elif pd.isnull(q3):
                q3 = 'NA'
            writer.writerow([img_url, worker_id, q1, q2, q3, q4])

def append_batch_results(batch_results_csv=COMBINED_BATCH_RESULTS_CSV,
                         mturk_results_csv=MTURK_RESULTS_CSV):
    with open(mturk_results_csv, 'ab') as mcsv, open(batch_results_csv, 'rb') as bcsv:
        reader = csv.reader(bcsv)
        next(reader)
        writer = csv.writer(mcsv, lineterminator='\n')
        for row in reader:
            writer.writerow(row)

####################
### Data loaders ###
####################

def sentences_df(sentence_csv=SENTENCES_CSV, labels='full', drop_unk=True,
                 label_unk=None, distant=None):
    """
    Create a dataframe out of the data in `sentence_csv`.
    Each row contains a sentence and its label. The label set
    is determined by the value of the `labels` parameter.
    """
    df = label_unk if isinstance(label_unk, pd.DataFrame) else pd.read_csv(sentence_csv)
    df = df.drop(['img_file'], 1)
    if labels == 'full':
        if drop_unk:
            df = df[df.q3 != 'other_unclear']
            df = df[df.q4 != 'other_unclear']
        def full_map(q1, q2, q3, q4):
            label = [q1, q2, q3, q4]
            label = [l for l in label if type(l) is str]
            return '/'.join(label)
        df['label'] = df.apply(
            lambda x: full_map(x['q1'], x['q2'], x['q3'], x['q4']), axis=1)
        df = df.drop(['q1', 'q2', 'q3', 'q4'], 1)
        return df
    elif labels == 'in_out':
        df = df.drop(['q2', 'q3', 'q4'], 1)
        df.columns = ['sentence', 'label']
        return df
    elif labels == 'man_nat':
        df = df.drop(['q1', 'q3', 'q4'], 1)
        df.columns = ['sentence', 'label']
        return df
    elif labels == '3way':
        def threeway_map(q1, q2, q3, q4):
            return '/'.join([q1, q2])
        df['label'] = df.apply(
            lambda x: threeway_map(x['q1'], x['q2'], x['q3'], x['q4']), axis=1)
        df = df.drop(['q1', 'q2', 'q3', 'q4'], 1)
        return df
    elif labels == 'function':
        if drop_unk:
            df = df[df.q3 != 'other_unclear']
            df = df[df.q4 != 'other_unclear']
        def fn_map(q1, q2, q3, q4):
            if isinstance(q3, str):
                return q3
            elif isinstance(q4, str):
                return 'natural'
        df['label'] = df.apply(
            lambda x: fn_map(x['q1'], x['q2'], x['q3'], x['q4']), axis=1)
        df = df.drop(['q1', 'q2', 'q3', 'q4'], 1)
        return df

def load_dataset(df, ngram_order=1, pad=False, stem=False, omit_stop=False):
    """
    Creates numpy arrays out of a dataframe. If `pad` is set to
    `True`, X array will be of size (n_samples, maxlen), where each
    element of a sample is an index corresponding to `word2idx`.
    Otherwise, X array will be of size (n_samples, vocab_size+1),
    where each element of a sample is either 1, indicating that
    the word corresponding to that index is in the sentence (bag
    of words/ngrams representation).
    """
    sentences = df['sentence'].values
    vocab = []
    stemmer = PorterStemmer()
    stop = stopwords.words('english')
    for sentence in sentences:
        if omit_stop:
            sentence = [w
                        for w in sentence.split()
                        if w not in stop and w not in "?.,-!"]
        else:
            sentence = [w
                        for w in sentence.split()
                        if w not in "?.,-!"]
        if ngram_order == 1:
            for word in sentence:
                if stem:
                    vocab.append(stemmer.stem(word))
                else:
                    vocab.append(word)
        else:
            for ngram in zip(*[sentence[i:] for i in range(ngram_order)]):
                vocab.append(ngram)
    # start at 1 to allow masking in Keras
    word2id = {w:i for i,w in enumerate(set(vocab), start=1)}
    X_ind = []
    for i,sentence in enumerate(sentences):
        if omit_stop:
            sentence = [w
                        for w in sentence.split()
                        if w not in stop and w not in "?.,-!"]
        else:
            sentence = [w
                        for w in sentence.split()
                        if w not in "?.,-!"]
        if ngram_order == 1:
            if stem:
                indices = [word2id[stemmer.stem(w)] for w in sentence]
            else:
                indices = [word2id[w] for w in sentence]
            X_ind.append(indices)
        else:
            indices = [word2id[n] for n in
                       zip(*[sentence[j:] for j in range(ngram_order)])]
    X = np.zeros((len(sentences), len(word2id)+1))
    for i,sample in enumerate(X_ind):
        for idx in sample:
            X[i,idx] = 1
    l_enc = LabelEncoder()
    y = l_enc.fit_transform(df['label'].values)
    if pad:
        X = pad_sequences(X_ind, maxlen=79)
    return X, y, word2id, l_enc

#######################
### Majority voting ###
#######################

def write_majority_vote_csv(results_csv, outfile):
    """
    Given the results of crowdsourced annotation with more than 1 assignment
    per HIT, perform majority voting and write the results to `outfile`.
    Does not write results with one or more questions without majority
    consensus.

    Parameters
    ----------
    results_csv: csv containing image urls and annotations (4 answers each)
        with responses from more than one annotator
    outfile: csv file to be written
    """
    datadict = make_datadict(results_csv)
    voted_datadict = majority_vote_dict(datadict, keep_all=False)
    with open(outfile, 'w') as out:
        header = ['image_url', 'worker_id', 'q1', 'q2', 'q3', 'q4']
        writer = csv.writer(out)
        writer.writerow(header)
        for img_file, answers in voted_datadict.items():
            row = list(answers)
            row.insert(0, 'majority') # worker_id
            row.insert(0, img_file)
            writer.writerow(row)

def majority_vote_dict(datadict, keep_all=True):
    """
    Performs majority voting on annotated data. For each question
    for each image file, select as the correct the answer given by
    two or more annotators. If there is no majority consensus,
    no answer will be selected as correct.

    Parameters
    ----------
    datadict: dictionary created by `datadict` method mapping each
        image filename to a list of lists where each sublist contains
        the annotators answers to each of the 4 questions asked in the
        task
    keep_all: boolean, if True, keep result for each image whether or
        not all of the corresponding questions have majority consensus;
        if False, discard annotations for images that contain at least
        one question without majority consensus from `voted_datadict`.

    Returns
    -------
    voted_datadict: dictionary mapping each image filename to either
        a list of 4 answers corresponding to the majority consensus
        for each question, or a list of lists corresponding to the
        original responses (if no consensus).
    """
    voted_datadict = {}
    # return (value, count) tuple for most common value
    most_common = lambda x: Counter(x).most_common(1)[0]
    is_majority = lambda x,n: x[1] >= max(2., ceil(n / 2.))

    nb_no_majority_imgs = 0
    nb_no_majority_questions = 0
    for img_file, answer_lists in datadict.items():
        if len(answer_lists) == 1:
            voted_datadict[img_file] = answer_lists[0]
        else:
            no_majority_img = False
            answers = zip(*[answer_list for answer_list in answer_lists])
            majority = [most_common(a) for a in answers]
            majority_answers = []
            for i, a in enumerate(majority):
                if is_majority(a, len(answers[i])):
                    majority_answers.append(a[0])
                else:
                    majority_answers.append(answers[i])
                    no_majority_img = True
                    nb_no_majority_questions += 1

            # check to make sure majorities for q3 and q4 both aren't NA
            is_na = lambda x: x == 'NA'
            if sum([is_na(m) for m in majority_answers]) > 1:
                no_majority_img = True
                majority_answers[0] = ('no', 'majority')
            # check to rule out invalid combos
            if majority_answers[1] == 'man-made' and \
                    majority_answers[3] in ['field_forest', 'body_of_water']:
                no_majority_img = True
                majority_answers[0] = ('invalid', 'data')

            if no_majority_img:
                #print "no majority: %s" % img_file
                nb_no_majority_imgs += 1
            voted_datadict[img_file] = majority_answers
            if not keep_all:
                voted_datadict = dict(voted_datadict)
                for img_file, answers in voted_datadict.items():
                    if any(isinstance(a, tuple) for a in answers):
                        del voted_datadict[img_file]
    print "No majority for %i questions/%i images." % \
        (nb_no_majority_questions, nb_no_majority_imgs)
    return voted_datadict

##################################
### Annotation result analysis ###
##################################

def unique_workers(results_csv):
    worker_ids = Counter()
    with open(results_csv) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            worker_ids[row[1]] += 1
    for k,v in worker_ids.most_common():
        print "%s: %s" % (k, v)
    return len(worker_ids)

def make_kappa_matrix(datadict, labels='full'):
    """
    Create a N x M matrix where N is the number of images
    and M is the number of labels (14 for full) to be used
    for computing fleiss's kappa
    """
    datadict = dict(datadict)
    if labels == 'full':
        def full_map(q1, q2, q3, q4):
            label = [q1, q2, q3, q4]
            label = [l for l in label if l != 'NA']
            return '/'.join(label)
        label_map = full_map
    elif labels == 'in_out':
        label_map = lambda q1,q2,q3,q4: q1
    elif labels == 'man_nat':
        label_map = lambda q1,q2,q3,q4: q2
    elif labels == '3way':
        def threeway_map(q1, q2, q3, q4):
            return '/'.join([q1, q2])
        label_map = threeway_map
    elif labels == 'function':
        def fn_map(q1, q2, q3, q4):
            if q3 == 'NA':
                return 'natural'
            else:
                return q3
        label_map = fn_map
    for img, answer_lists in datadict.items():
        datadict[img] = [label_map(*a) for a in answer_lists]
        if len(datadict[img]) < 3:
            del datadict[img]
    labels = set([a for answer_list in datadict.values() for a in answer_list])
    label2id = {l:i for i,l in enumerate(labels)}
    nb_labels = len(labels)
    matrix = np.zeros((len(datadict), nb_labels), dtype='int32')
    for i, (img, label_lists) in enumerate(datadict.items()):
        for label in label_lists:
            matrix[i, label2id[label]] += 1
    return matrix

def fleiss_kappa(results_csv, labels='full'):
    from scripts import fleiss
    datadict = make_datadict(results_csv)
    matrix = make_kappa_matrix(datadict, labels=labels)
    return fleiss.kappa(matrix)

def write_no_majority_list():
    with open(REDO_IMGS_FILE, 'a') as r:
        datadict = make_datadict(COMBINED_MTURK_RESULTS_CSV, keep_url=True)
        voted_dict = majority_vote_dict(datadict, keep_all=True)
        for img_file, answers in voted_dict.items():
            if any(isinstance(a, tuple) for a in answers):
                r.write(img_file + '\n')

########################
### Helper functions ###
########################

def combine_csvs(csv1, csv2, outcsv):
    with open(csv1,'rb') as c1, open(csv2,'rb') as c2, open(outcsv, 'wb') as outfile:
        c1reader = csv.reader(c1)
        c2reader = csv.reader(c2)
        next(c1reader)
        next(c2reader)
        writer = csv.writer(outfile)
        for row in c1reader:
            writer.writerow(row)
        for row in c2reader:
            writer.writerow(row)
    print "Combined %s and %s into %s." % (basename(csv1),
                                           basename(csv2),
                                           basename(outcsv))

def url2filename(url):
    return url.split('/')[-1]

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec. Taken from
    CNN_sentence https://github.com/yoonkim/CNN_sentence
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len),
                                               dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, k=300):
    unknown_words = []
    for word in vocab:
        if word not in word_vecs:
            unknown_words.append(word)
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
    word_vecs['<unk>'] = np.random.uniform(-0.25,0.25,k)
    print "Added %i unknown words to word vectors." % len(unknown_words)
    #print unknown_words
