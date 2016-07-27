import numpy as np
import time
import sys
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, GRU
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.utils.layer_utils import print_summary
from sklearn.cross_validation import StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.data_utils import (
    sentences_df,
    load_dataset,
    load_bin_vec
)
from paths import SENTENCES_CSV


def add_unknown_words(word_vecs, vocab, k=300):
    added = 0
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
            added += 1
    print "Added %i unknown words to word vectors." % added

def create_model(n_vocab, n_labels, vocab_dim,
                 embedding_weights, rnn_layer='lstm'):
    model = Sequential()
    model.add(Embedding(input_dim=n_vocab+1, output_dim=300,
                        mask_zero=True, dropout=0.2,
                        weights=[embedding_weights]))
    if rnn_layer == 'lstm':
        model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2,
                       return_sequences=False))
    elif rnn_layer == 'gru':
        model.add(GRU(512, dropout_W=0.5, dropout_U=0.5,
                      return_sequences=True, activation='relu'))
        model.add(GRU(512, dropout_W=0.5, dropout_U=0.5,
                      return_sequences=False, activation='relu'))
    model.add(Dense(n_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train_and_test_model(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0,
                                   mode='auto')
    model.fit(X_train, y_train, batch_size=128, nb_epoch=8)
    score = model.evaluate(X_test, y_test, batch_size=64)
    return score

def main(rnn_layer='lstm', word_vecs=None):
    print "Loading data...",
    df = sentences_df(SENTENCES_CSV)
    X, y, word2idx, l_enc = load_dataset(df, pad=True)
    print X.shape
    y_binary = to_categorical(y)
    word_vectors = load_bin_vec(word_vecs, word2idx)
    add_unknown_words(word_vectors, word2idx)
    print "Data loaded."

    labels = np.unique(y)
    n_labels = labels.shape[0]
    max_len = X.shape[1]
    vocab_dim = 300
    n_vocab = len(word2idx) + 1 # 0 masking
    embedding_weights = np.zeros((n_vocab+1, vocab_dim))
    for word, index in word2idx.items():
        embedding_weights[index,:] = word_vectors[word]

    skf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=0)
    cv_scores = []
    for i, (train, test) in enumerate(skf):
        start_time = time.time()
        model = create_model(n_vocab, n_labels, vocab_dim,
                             embedding_weights, rnn_layer=rnn_layer)
        if i == 0:
            print_summary(model.layers)
        _, score = train_and_test_model(model, X[train], y_binary[train],
                                     X[test], y_binary[test])
        cv_scores.append(score)
        train_time = time.time() - start_time
        print "fold %i/10 - time: %.2f s - acc: %.4f on %i samples" % \
            (i+1, train_time, score, len(test))
    print "avg cv acc: %.4f" % np.mean(cv_scores)

if __name__ == '__main__':
    main(rnn_layer=sys.argv[1], word_vecs=sys.argv[2])
