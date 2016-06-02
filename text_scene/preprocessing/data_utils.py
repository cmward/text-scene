import csv
import numpy as np
import pandas as pd
from math import ceil
from os.path import basename
from collections import defaultdict, Counter
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
    COMBINED_MTURK_RESULTS_CSV
)

q1map = {'0': 'indoors', '1': 'outdoors'}
q2map = {'0': 'man-made', '1': 'natural'}
q3map = {'0': 'transportation_urban',
         '1': 'restaurant',
         '2': 'recreation_entertainment',
         '3': 'domestic',
         '4': 'work_education',
         '5': 'athletics',
         '6': 'shop',
         '7': 'other_unclear',
         'NA': 'NA'}
q4map = {'0': 'body_of_water',
         '1': 'field',
         '2': 'mountain',
         '3': 'forrest_jungle',
         '4': 'other_unclear',
         'NA': 'NA'}

def url2filename(url):
    return url.split('/')[-1]

def make_datadict(results_csv, keep_url=False):
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
    with open(captions_file) as cfile, open(out_csv, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['sentence', 'q1', 'q2', 'q3', 'q4', 'img_file'])
        n_sents = 0
        for line in cfile:
            split_line = line.split()
            img_file = split_line[0].split('#')[0]
            if img_file in datadict:
                n_sents += 1
                sentence = ' '.join(split_line[1:]).lower()
                annotations = datadict[img_file]
                if len(annotations) != 4:
                    annotations = annotations[0]
                assert len(annotations) == 4
                writer.writerow([sentence] + annotations + [img_file])
    print "Wrote sentence csv with %i sentences." % n_sents

def load_data(sentence_csv, labels='full', drop_unk=True):
    """
    Create a dataframe out of the data in `sentence_csv`.
    Each row contains a sentence and its label. The label set
    is determined by the value of the `labels` parameter.
    """
    df = pd.read_csv(sentence_csv)
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

def load_dataset(df, ngram_order=1, pad=False):
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
    for sentence in sentences:
        sentence = sentence.split()
        if ngram_order == 1:
            for word in sentence:
                vocab.append(word)
        else:
            for ngram in zip(*[sentence[i:] for i in range(ngram_order)]):
                vocab.append(ngram)
    # start at 1 to allow masking in Keras
    word2id = {w:i for i,w in enumerate(set(vocab), start=1)}
    X_ind = []
    for i,sentence in enumerate(sentences):
        sentence = sentence.split()
        if ngram_order == 1:
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
        X = pad_sequences(X_ind)
    return X, y, word2id, l_enc

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
            if no_majority_img:
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

def unique_workers(results_csv):
    worker_ids = set()
    with open(results_csv) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            worker_ids.add(row[1])
    return len(worker_ids)

def make_kappa_matrix(datadict, labels='full'):
    """
    Create a N x M matrix where N is the number of images
    and M is the number of labels (21 for full) to be used
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

def write_rejected_no_majority_list():
    with open(REDO_IMGS_FILE, 'a') as r:
        datadict = make_datadict(COMBINED_MTURK_RESULTS_CSV, keep_url=True)
        voted_dict = majority_vote_dict(datadict, keep_all=True)
        for img_file, answers in voted_dict.items():
            if any(isinstance(a, tuple) for a in answers):
                r.write(img_file + '\n')
        with open(REJECTED_IMGS_FILE, 'r') as f:
            for line in f:
                r.write(line.strip() + '\n')

def restore_rejected_imgs():
    """
    Remove rejected images from annotated images file and
    then remove them from rejected images file.
    """
    all_images = set()
    annotated = set()
    rejected = set()
    with open(IMG_URLS, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            all_images.add(row[0])
    with open(ANNOTATED_IMGS_FILE, 'r') as a:
        for line in a:
            annotated.add(line.strip())
    with open(REJECTED_IMGS_FILE, 'r') as r:
        for line in r:
            rejected.add(line.strip())
    no_rejects_annotated = annotated - rejected
    with open(ANNOTATED_IMGS_FILE, 'w') as a:
        for img in no_rejects_annotated:
            a.write(img + '\n')
    open(REJECTED_IMGS_FILE, 'w').close()

def combine_csvs(csv1, csv2, outcsv):
    with open(csv1) as c1, open(csv2) as c2, open(outcsv, 'w') as outfile:
        c1reader = csv.reader(c1)
        c2reader = csv.reader(c2)
        next(c2reader)
        writer = csv.writer(outfile)
        for row in c1reader:
            writer.writerow(row)
        for row in c2reader:
            writer.writerow(row)
    print "Combined %s and %s into %s." % (basename(csv1),
                                           basename(csv2),
                                           basename(outcsv))

def label_frequencies(data):
    """
    Given a dataset, return a dictionary mapping labels to label frequencies.

    `data` is either a single dataframe or a tuple (y, l_enc) where y is a
    numpy array of labels and l_enc is a LabelEncoder.
    """
    if isinstance(data, pd.DataFrame):
        counts = defaultdict(float)
        for i, row in data.iterrows():
            counts[row['label']] += 1
    elif isinstance(data, tuple):
        y = data[0]
        l_enc = data[1]
        bincounts = np.bincount(y)
        counts = {l_enc.inverse_transform(i): float(count)
                 for i, count in enumerate(bincounts)}
    total = sum(counts.values())
    freqs = {k: (v/total) for k,v in counts.items()}
    sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    return sorted_freqs

def print_label_frequencies(data):
    freqs = label_frequencies(data)
    for (label, freq) in freqs:
        print "%s: %.2f" % (label, freq)
