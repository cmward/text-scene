import numpy as np
import time
import sys
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, GRU
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import load_data
from CNN_sentence.process_data import load_bin_vec
from paths import SENTENCES_CSV


def add_unknown_words(word_vecs, vocab, k=300):
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def create_model(n_vocab, n_labels, vocab_dim,
                 embedding_weights, rnn_layer='lstm'):
    model = Sequential()
    model.add(Embedding(input_dim=n_vocab+1, output_dim=300, mask_zero=True,
                        weights=[embedding_weights]))
    if rnn_layer == 'lstm':
        model.add(LSTM(128, return_sequences=False))
    elif rnn_layer == 'gru':
        model.add(GRU(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(n_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def train_and_test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, batch_size=16, nb_epoch=25)
    score = model.evaluate(X_test, y_test, batch_size=16)
    return score

def main(word_vecs, rnn_layer='lstm'):
    print "Loading data...",
    df = load_data.load_data(SENTENCES_CSV)
    X, y, word2idx, l_enc = load_data.load_dataset(df, pad=True)
    y_binary = to_categorical(y)
    word_vectors = load_bin_vec(word_vecs, word2idx)
    add_unknown_words(word_vectors, word2idx)
    print "Data loaded."

    labels = np.unique(y)
    n_labels = labels.shape[0]
    max_len = 82
    vocab_dim = 300
    n_vocab = len(word2idx) + 1 # 0 masking
    embedding_weights = np.zeros((n_vocab+1, vocab_dim))
    for word, index in word2idx.items():
        embedding_weights[index,:] = word_vectors[word]

    skf = StratifiedKFold(y, n_folds=10, shuffle=True)
    for i, (train, test) in enumerate(skf):
        start_time = time.time()
        model = create_model(n_vocab, n_labels, vocab_dim,
                             embedding_weights, rnn_layer=rnn_layer)
        score = train_and_test_model(model, X[train], y_binary[train],
                                     X[test], y_binary[test])
        train_time = time.time() - start_time
        print "fold %i/10 - time: %.2f - acc: %.2f" % (i,train_time,score)

if __name__ == '__main__':
    main(sys.argv[1], rnn_layer=sys.argv[2])
