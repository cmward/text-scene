import re
import json
import os
import sys
import numpy as np
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
captions_file = sys.argv[2]

df = sentences_df(labels='full', drop_unk=True)
X, y, word2idx, l_enc = load_dataset(df, ngram_order=1, pad=True)

elc_sents = set(list(df.sentence))
flickr_sents = []
with open('../../results_20130124.token') as f:
    for line in f:
        line = line.strip().lower().split('\t')[-1]
        if line not in elc_sents:
            flickr_sents.append(line)

idx = len(word2idx) + 1
for sent in flickr_sents:
    for word in re.split("-| ", sent):
        if word not in word2idx:
            word2idx[word] = idx
            idx += 1

#with open('ae_word2idx.json','r') as j:
#    word2idx = json.load(j)

X_f_indices = []
for sent in flickr_sents:
    indices = [word2idx[word]
               for word in re.split("-| ", sent)]
    X_f_indices.append(indices)
X_f = pad_sequences(X_f_indices, maxlen=78, padding='post')

print len(X_f), len(word2idx)

train_split = 100000
X_train = X_f[:train_split]
X_test = X_f[:train_split]

def minibatches(X_f, word2idx, batch_size=64):
    nb_samples = X_f.shape[0]
    nb_batches = nb_samples // batch_size
    while True:
        for batch_idx in range(nb_batches):
            X = X_f[batch_idx * batch_size : (batch_idx+1) * batch_size]
            y = np.asarray([to_categorical(x_i, nb_classes=len(word2idx)+1)
                            for x_i in X]).astype(np.int32)
            yield X, y

vocab_size = len(word2idx) + 1
emb_dim = 300
maxlen = 78

word_vectors = load_bin_vec(word_vecs, word2idx)
add_unknown_words(word_vectors, word2idx)
embedding_weights = np.zeros((vocab_size,300))
for word, index in word2idx.items():
    embedding_weights[index:,] = word_vectors[word]

with open('ae_word2idx.json','w') as j:
    json.dump(word2idx, j)

autoencoder = Sequential()
autoencoder.add(Embedding(input_dim=vocab_size,
                 output_dim=emb_dim,
                 input_length=maxlen,
                 weights=[embedding_weights]))
autoencoder.add(Dropout(0.5))

autoencoder.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))
autoencoder.add(Dropout(0.5))

autoencoder.add(Dense(512))
autoencoder.add(BatchNormalization())
autoencoder.add(PReLU())
autoencoder.add(Dropout(0.5))

autoencoder.add(RepeatVector(maxlen))
autoencoder.add(TimeDistributed(Dense(300)))
autoencoder.add(BatchNormalization())
autoencoder.add(PReLU())
autoencoder.add(Dropout(0.5))

autoencoder.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

autoencoder.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

print_summary(autoencoder.layers)

generator = minibatches(X_train, word2idx)
samples_per_epoch = X_train.shape[0]
autoencoder.fit_generator(generator,
                          samples_per_epoch=samples_per_epoch,
                          nb_epoch=2)

autoencoder.save_weights('autoencoder.h5')
