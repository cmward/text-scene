import csv
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from paths import CAPTIONS_FILE, SENTENCES_CSV

q1map = {'0': 'indoors', '1': 'outdoors'}
q2map = {'0': 'man-made', '1': 'natural'}
q3map = {'0': 'transportation_urban',
         '1': 'food_drink',
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

def make_datadict(csvfile):
    datadict = {}
    with open(csvfile, 'r') as csvf:
        reader = csv.reader(csvf)
        next(reader)
        for row in reader:
            img_file = url2filename(row[0])
            q1 = q1map[row[2]]
            q2 = q2map[row[3]]
            q3 = q3map[row[4]]
            q4 = q4map[row[5]]
            datadict[img_file] = [q1, q2, q3, q4]
    return datadict

def write_sentence_csv(datadict, captions_file, out_csv):
    sentences = []
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    with open(captions_file) as cfile, open(out_csv, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['sentence', 'q1', 'q2', 'q3', 'q4', 'img_file'])
        for line in cfile:
            split_line = line.split()
            img_file = split_line[0].split('#')[0]
            if img_file in datadict:
                sentence = ' '.join(split_line[1:]).lower()
                annotations = datadict[img_file]
                writer.writerow([sentence] + annotations + [img_file])

def load_data(sentence_csv, labels='full'):
    df = pd.read_csv(sentence_csv)
    df = df.drop(['img_file'], 1)
    if labels == 'full':
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
    elif labels == 'function':
        def fn_map(q1, q2, q3, q4):
            if isinstance(q3, str):
                return q3
            elif isinstance(q4, str):
                return 'natural'
        df['label'] = df.apply(
            lambda x: fn_map(x['q1'], x['q2'], x['q3'], x['q4']), axis=1)
        df = df.drop(['q1', 'q2', 'q3', 'q4'], 1)
        return df

def load_dataset(df, ngram_order=1):
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
    word2id = {w:i for i,w in enumerate(set(vocab))}
    X_ind = []
    for i,sentence in enumerate(sentences):
        sentence = sentence.split()
        if ngram_order == 1:
            indices = [word2id[w] for w in sentence]
            X_ind.append(indices)
        else:
            indices = [word2id[n] for n in
                       zip(*[sentence[j:] for j in range(ngram_order)])]
    X = np.zeros((len(sentences), len(word2id)))
    for i,sample in enumerate(X_ind):
        for idx in sample:
            X[i,idx] = 1
    l_enc = LabelEncoder()
    y = l_enc.fit_transform(df['label'].values)
    return X, y, word2id, l_enc
