import numpy as np
import re
import json
import argparse
import sys
import os

from keras.models import Model
from keras.layers import Input, Dense, LSTM, merge, Embedding
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from classification.cnn import ParallelCNN, max_1d
from preprocessing.data_utils import (
    load_bin_vec,
    sentences_df,
    load_dataset,
    add_unknown_words
)

def load_from_json(json_file):
    samples = []
    with open(json_file) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def preprocess(sent):
    # sent is single string
    # return tokens
    punct = "?.,-;!"
    sent = sent.lower()
    tokens = [token
              for token in re.split("-| ", sent)
              if token not in punct]
    tokens = [token
              if not token[-1] in punct
              else token[:-1]
              for token in tokens]
    return tokens

def build_vocab(samples):
    # samples is [train_samples, val_samples, test_samples]
    all_sent1 = set(sample['sentence1']
                    for split_samples in samples
                    for sample in split_samples)
    all_sent2= set(sample['sentence2']
                    for split_samples in samples
                    for sample in split_samples)
    all_sents = all_sent1.union(all_sent2)

    vocab = set(token
                for sentence in all_sents
                for token in preprocess(sentence))
    word2idx = {w:i
                for i,w in enumerate(vocab)}

    return vocab, word2idx

def remove_overlap(samples, text_scene_df):
    overlap = lambda sample, files: (sample['captionID'].split('#')[0]
                                     in files)
    text_scene_files = set(text_scene_df.img_file)
    filtered_samples = [sample
                        for sample in samples
                        if not overlap(sample, text_scene_files)]
    return filtered_samples

def vectorize(samples, word2idx, l_enc, text_scene_df, maxlen,
              rm_overlap=False):
    # samples is one of {train,test,val}_samples
    # X is list of 2 arrays, sentence1, sentence2
    # y is gold label

    # get rid of test/val samples where one of the sentences occurs
    # in text-scene, so we're not training on test data
    if rm_overlap:
        samples = remove_overlap(samples, text_scene_df)
    prem_indices = [[word2idx[word] for word in preprocess(sample['sentence1'])]
                   for sample in samples]
    hyp_indices = [[word2idx[word] for word in preprocess(sample['sentence2'])]
                    for sample in samples]
    prem = pad_sequences(prem_indices, maxlen=maxlen)
    hyp = pad_sequences(hyp_indices, maxlen=maxlen)
    X = [prem, hyp]

    try:
        y = l_enc.transform([sample['gold_label']
                             for sample in samples])
    except:
        y = l_enc.fit_transform([sample['gold_label']
                                 for sample in samples])

    return X, y, l_enc

def build_model(maxlen, vocab_size, word2idx, embeddings_path, nb_labels,
                sent_model='lstm', ffweights=None):
    # load word embeddings
    word_vectors = load_bin_vec(embeddings_path, word2idx)
    add_unknown_words(word_vectors, word2idx)
    embedding_weights = np.zeros((vocab_size+1, 300))
    for word, index in word2idx.items():
        embedding_weights[index,:] = word_vectors[word]

    prem = Input(shape=(maxlen,), dtype='int32')
    hyp = Input(shape=(maxlen,), dtype='int32')


    if sent_model == 'lstm':
        emb = Embedding(input_dim=vocab_size+1, output_dim=300,
                        input_length=maxlen, weights=[embedding_weights])
        prem_emb = emb(prem)
        hyp_emb = emb(hyp)

        encoder = LSTM(100, activation='relu', return_sequences=False)
        prem_enc = encoder(prem_emb)
        hyp_enc = encoder(hyp_emb)

    else:
        sentence_input = Input(shape=(maxlen,), dtype='int32')
        x = Embedding(input_dim=vocab_size+1, output_dim=300,
                      input_length=maxlen,
                      weights=[embedding_weights])
        x = x(sentence_input)

        if ffweights:
            model.load_weights(ffweights)

        prem_emb = x(prem)
        hyp_emb = x(hyp)

        prem_enc = ff(prem_emb)
        hyp_enc = ff(hyp_emb)

    merged = merge([prem_enc, hyp_enc], mode='concat', concat_axis=-1)
    dropout = Dropout(0.5)(merged)

    fc1 = Dense(200, activation='relu')(merged_sents)
    fc2 = Dense(200, activation='relu')(fc1)
    fc3 = Dense(200, activation='relu')(fc2)
    probas = Dense(nb_labels, activation='softmax')(fc3)

    model = Model(input=[prem, hyp], output=probas)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', type=str)
    argparser.add_argument('--val', type=str)
    argparser.add_argument('--test', type=str)
    argparser.add_argument('--wordvecs', type=str)
    argparser.add_argument('--sentmodel', type=str)
    args = argparser.parse_args()

    print "Loading data... ",
    df = sentences_df(labels='full', drop_unk=True, keep_filename=True)
    train_samples = load_from_json(args.train)
    val_samples = load_from_json(args.val)
    test_samples = load_from_json(args.test)
    samples = [train_samples, val_samples, test_samples]
    vocab, word2idx = build_vocab(samples)
    vocab_size = len(vocab)
    maxlen = max(max(len(preprocess(sample['sentence1'])),
                     len(preprocess(sample['sentence2'])))
                 for split in samples
                 for sample in split)
    l_enc = LabelEncoder()

    X_train, y_train, l_enc = vectorize(train_samples, word2idx, l_enc, df, maxlen)
    X_val, y_val, _ = vectorize(val_samples, word2idx, l_enc, df, maxlen,
                                rm_overlap=True)
    X_test, y_test, _ = vectorize(test_samples, word2idx, l_enc, df, maxlen,
                                  rm_overlap=True)
    nb_labels = len(l_enc.classes_)
    print "Loaded."

    print "Building model...",
    model = build_model(maxlen, vocab_size, word2idx, args.wordvecs,
                        nb_labels, sent_model=args.sentmodel)
    model.fit(X_train, y_train)
