import numpy as np
import time
import sys
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Lambda, Merge
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils.layer_utils import print_summary
from keras import backend as K
from sklearn.cross_validation import StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import load_data
from CNN_sentence.process_data import load_bin_vec
from paths import SENTENCES_CSV


def add_unknown_words(word_vecs, vocab, k=300):
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def max_1d(X):
    return K.max(X, axis=1)

def DeepCNN(n_vocab, n_labels, vocab_dim, maxlen, embedding_weights):
    model = Sequential()
    model.add(Embedding(input_dim=n_vocab+1,
                        output_dim=300,
                        input_length=maxlen,
                        dropout=0.2,
                        weights=[embedding_weights]))
    model.add(Convolution1D(128, 2, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Convolution1D(64, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Convolution1D(32, 4, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    if n_labels == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.add(Dense(n_labels, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model, []

def ParallelCNN(n_vocab, n_labels, vocab_dim, maxlen, embedding_weights):
    filter_hs = [3,4,5]
    n_filters = 10
    model = Sequential()
    submodels = []
    for filter_h in filter_hs:
        submodel = Sequential()
        submodel.add(Embedding(input_dim=n_vocab+1,
                               output_dim=300,
                               input_length=maxlen,
                               dropout=0.2,
                               weights=[embedding_weights]))
        submodel.add(Convolution1D(n_filters, filter_h,
                                   border_mode='same',
                                   activation='relu'))
        submodel.add(Lambda(max_1d, output_shape=(n_filters,)))
        submodels.append(submodel)
    model.add(Merge(submodels, mode='concat'))
    if n_labels == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.add(Dense(n_labels, activation='softmax'))
        model.add(Dropout(0.2))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model, filter_hs

def create_model(n_vocab, n_labels, vocab_dim, maxlen,
                 embedding_weights, model_type='parallel'):
    """ Create CNN with architecture specified by `model_type.`
    Returns the model and filter heights."""
    if model_type == 'deep':
        model, filter_hs = DeepCNN(n_vocab, n_labels, vocab_dim,
                                   maxlen, embedding_weights)
    elif model_type == 'parallel':
        model, filter_hs = ParallelCNN(n_vocab, n_labels, vocab_dim,
                                       maxlen, embedding_weights)
    return model, filter_hs

def train_and_test_model(model, model_type, filter_hs,
                         X_train, y_train, X_test, y_test):
    if model_type == 'parallel':
        model.fit([X_train]*len(filter_hs), y_train,
                  batch_size=32, nb_epoch=7,
                  validation_split=0.2)
        score = model.evaluate([X_test]*len(filter_hs), y_test, batch_size=32)
    else:
        model.fit(X_train, y_train,
                  batch_size=32, nb_epoch=7,
                  validation_split=0.2)
        score = model.evaluate(X_test, y_test, batch_size=32)
    return score

def main(model_type='parallel', label_set='full', setup_only=False):
    print "Loading data...",
    df = load_data.load_data(SENTENCES_CSV, labels=label_set)
    X, y, word2idx, l_enc = load_data.load_dataset(df, pad=True)
    y_orig = y
    y_binary = to_categorical(y)
    word_vectors = load_bin_vec(
        '../../data/GoogleNews-vectors-negative300.bin', word2idx)
    add_unknown_words(word_vectors, word2idx)
    print "Data loaded."

    labels = np.unique(y)
    n_labels = labels.shape[0]
    print "Number of labels:", n_labels
    if n_labels > 2:
        y = y_binary
    maxlen = X.shape[1]
    vocab_dim = 300
    n_vocab = len(word2idx) + 1 # 0 masking
    embedding_weights = np.zeros((n_vocab+1, vocab_dim))
    for word, index in word2idx.items():
        embedding_weights[index,:] = word_vectors[word]

    if setup_only:
        return {'X': X,
                'y': y,
                'word2idx': word2idx,
                'l_enc': l_enc,
                'y_binary': y_binary,
                'word_vectors': word_vectors,
                'labels': labels,
                'n_labels': n_labels,
                'maxlen': maxlen,
                'vocab_dim': vocab_dim,
                'n_vocab': n_vocab,
                'embedding_weights': embedding_weights}

    skf = StratifiedKFold(y_orig, n_folds=10, shuffle=True, random_state=0)
    cv_scores = []
    for i, (train, test) in enumerate(skf):
        start_time = time.time()
        model = None
        model, filter_hs = create_model(n_vocab,
                                        n_labels,
                                        vocab_dim,
                                        maxlen,
                                        embedding_weights,
                                        model_type=model_type)
        if i == 0:
            print_summary(model.layers)
        scores = train_and_test_model(model, model_type, filter_hs,
                                      X[train], y[train],
                                      X[test], y[test])
        cv_scores.append(scores[1])
        train_time = time.time() - start_time
        if n_labels == 2:
            y_1_perc = np.where(y[test]==1)[0].shape[0] / float(y[test].shape[0])
            print "y test labels == 1: %.2f" % y_1_perc
        print "fold %i/10 - time: %.2f - acc: %.2f" % (i+1,train_time,scores[1])
    print "Avg cv accuracy: %.2f" % np.mean(cv_scores)

if __name__ == '__main__':
    main(model_type=sys.argv[1], label_set=sys.argv[2])
