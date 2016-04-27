import os
import sys
import time
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Lambda, Merge, merge
from keras.layers import Embedding, Input
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.utils.layer_utils import print_summary
from keras import backend as K
from sklearn.cross_validation import StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_utils import load_data, load_dataset
from CNN_sentence.process_data import load_bin_vec
from paths import SENTENCES_CSV


def add_unknown_words(word_vecs, vocab, k=300):
    added = 0
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
            added += 1
    print "Added %i unknown words to word vectors." % added

def max_1d(X):
    return K.max(X, axis=1)

def DeepCNN(n_vocab, n_labels, emb_dim, maxlen, embedding_weights):
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
    return model

def ParallelCNN(n_vocab, n_labels, emb_dim, maxlen, embedding_weights):
    filter_hs = [3,4,5]
    n_filters = 16
    sentence_input = Input(shape=(maxlen,), dtype='int32')
    x = Embedding(input_dim=n_vocab+1, output_dim=emb_dim, input_length=maxlen,
                  dropout=0.2, weights=[embedding_weights])
    x = x(sentence_input)
    conv_pools = []
    for filter_h in filter_hs:
        conv = Convolution1D(n_filters, filter_h, border_mode='same',
                             activation='relu')
        conved = conv(x)
        pool = Lambda(max_1d, output_shape=(n_filters,))
        pooled = pool(conved)
        conv_pools.append(pooled)
    merged = merge(conv_pools, mode='concat')
    dropout = Dropout(0.5)(merged)
    if n_labels == 2:
        out = Dense(1, activation='sigmoid')(dropout)
        model = Model(input=sentence_input, output=out)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        out = Dense(n_labels, activation='softmax')(dropout)
        model = Model(input=sentence_input, output=out)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model

def create_model(n_vocab, n_labels, emb_dim, maxlen,
                 embedding_weights, model_type='parallel'):
    """ Create CNN with architecture specified by `model_type.`"""
    if model_type == 'deep':
        model = DeepCNN(n_vocab, n_labels, emb_dim, maxlen,
                        embedding_weights)
    elif model_type == 'parallel':
        model = ParallelCNN(n_vocab, n_labels, emb_dim, maxlen,
                            embedding_weights)
    return model

def train_and_test_model(model, model_type, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, batch_size=64, nb_epoch=15, validation_split=0.2)
    score, acc = model.evaluate(X_test, y_test, batch_size=64)
    return acc

def main(model_type='parallel', label_set='full',
         word_vecs=None, setup_only=False):
    print "Loading data...",
    df = load_data(SENTENCES_CSV, labels=label_set)
    X, y, word2idx, l_enc = load_dataset(df, pad=True)
    y_orig = y
    y_binary = to_categorical(y)
    word_vectors = load_bin_vec(word_vecs, word2idx)
    add_unknown_words(word_vectors, word2idx)
    print "Data loaded."

    labels = np.unique(y)
    n_labels = labels.shape[0]
    print "Number of labels:", n_labels
    if n_labels > 2:
        y = y_binary
    maxlen = X.shape[1]
    emb_dim = 300
    n_vocab = len(word2idx) + 1 # 0 masking
    embedding_weights = np.zeros((n_vocab+1, emb_dim))
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
                'emb_dim': emb_dim,
                'n_vocab': n_vocab,
                'embedding_weights': embedding_weights}

    skf = StratifiedKFold(y_orig, n_folds=10, shuffle=True, random_state=0)
    cv_scores = []
    for i, (train, test) in enumerate(skf):
        start_time = time.time()
        model = None
        model = create_model(n_vocab,
                             n_labels,
                             emb_dim,
                             maxlen,
                             embedding_weights,
                             model_type=model_type)
        if i == 0:
            print_summary(model.layers)
        acc = train_and_test_model(model, model_type,
                                   X[train], y[train],
                                   X[test], y[test])
        cv_scores.append(acc)
        train_time = time.time() - start_time
        if n_labels == 2:
            y_1 = np.where(y[test]==1)[0].shape[0]
            y_1_perc =  y_1 / float(y[test].shape[0])
            print "y_test[label == 1]: %2f%%" % y_1_perc
        print "fold %i/10 - time: %.2f - acc: %.2f on %i samples" % \
            (i+1, train_time, acc, len(test))
    print "Avg cv accuracy: %.2f" % np.mean(cv_scores)

if __name__ == '__main__':
    main(model_type=sys.argv[1],
         label_set=sys.argv[2],
         word_vecs=sys.argv[3])
