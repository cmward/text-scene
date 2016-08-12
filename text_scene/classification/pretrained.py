import re
import json
import os
import sys
import time
import h5py
import numpy as np

from sklearn.cross_validation import StratifiedKFold

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, TimeDistributed
from keras.layers import Flatten, Lambda, Merge, merge, RepeatVector
from keras.layers import Embedding, Input, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.utils.layer_utils import print_summary

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.data_utils import (
    load_bin_vec,
    sentences_df,
    add_unknown_words,
    load_dataset,
)


word_vecs = sys.argv[1]
weights_path = sys.argv[2]

# hyperparameters
emb_dim = 300
batch_size = 128
nb_epoch = 15
lr = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08


df = sentences_df(labels='full', drop_unk=True)
X, y, word2idx, l_enc = load_dataset(df, ngram_order=1, pad=True)

y_orig = y
y_binary = to_categorical(y)
labels = np.unique(y_orig)
nb_labels = labels.shape[0]
if nb_labels > 2:
    y = y_binary
maxlen = X.shape[1]

with open('ae_word2idx.json') as j:
    word2idx = json.load(j)
vocab_size = len(word2idx) + 1 # 0 masking
word_vectors = load_bin_vec(word_vecs, word2idx)
add_unknown_words(word_vectors, word2idx)

embedding_weights = np.zeros((vocab_size, emb_dim))
for word, index in word2idx.items():
    embedding_weights[index,:] = word_vectors[word]

idx2word = {i:w for w,i in word2idx.items()}
randidx = np.random.randint(low=1, high=len(idx2word)-1)
assert np.all(embedding_weights[randidx] ==
              word_vectors[idx2word[randidx]])
print "Data loaded."


def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=emb_dim,
                        input_length=maxlen,
                        weights=[embedding_weights]))
    model.add(Dropout(0.2))

    model.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(RepeatVector(maxlen))
    model.add(TimeDistributed(Dense(300)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    model.load_weights(weights_path)
    for _ in range(6):
        model.pop()
    model.add(Dense(nb_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

skf = StratifiedKFold(y_orig, n_folds=10, shuffle=True, random_state=0)
cv_scores = []
for i, (train, test) in enumerate(skf):
    start_time = time.time()
    nn = None
    nn = build_model()
    if i == 0:
        print_summary(nn.layers)
    nn.fit(X[train], y[train], nb_epoch=nb_epoch, batch_size=batch_size,
           validation_split=0.00)
    score, acc = nn.evaluate(X[test], y[test], batch_size=64)
    cv_scores.append(acc)
    train_time = time.time() - start_time
    print "\nfold %i/10 - time: %.2f s - acc: %.4f on %i samples" % \
            (i+1, train_time, acc, len(test))
print "Avg cv accuracy: %.4f" % np.mean(cv_scores)
