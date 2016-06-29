import os
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Lambda, Merge, merge
from keras.layers import Embedding, Input, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K

class FeedforwardNN(object):
    def __init__(self, vocab_size, nb_labels, emb_dim, maxlen, layer_sizes,
                 embedding_weights, pool_mode='max'):
        self.nb_labels = nb_labels
        sentence_input = Input(shape=(maxlen,), dtype='int32')
        x = Embedding(input_dim=vocab_size+1,
                      output_dim=emb_dim,
                      input_length=maxlen,
                      weights=[embedding_weights],
                      dropout=0.2)
        x = x(sentence_input)
        if pool_mode == 'sum':
            pool = Lambda(lambda x: K.sum(x, axis=1), output_shape=(emb_dim,))
        elif pool_mode == 'max':
            pool = Lambda(lambda x: K.max(x, axis=1), output_shape=(emb_dim,))
        elif pool_mode == 'mean':
            pool = Lambda(lambda x: K.mean(x, axis=1), output_shape=(emb_dim,))
        else: # concat
            pool = Flatten()
        pool_out = Dropout(0.5)(pool(x))
        hidden_layers = []
        prev_layer = pool_out
        for layer_size in layer_sizes:
            hidden_in = Dense(layer_size)(prev_layer)
            hidden_bn = BatchNormalization()(hidden_in)
            hidden_activation = LeakyReLU()(hidden_bn)
            hidden_out = Dropout(0.5)(hidden_activation)
            hidden_layers.append(hidden_out)
            prev_layer = hidden_out
        if self.nb_labels == 2:
            out = Dense(1, activation='sigmoid')
        else:
            out = Dense(nb_labels, activation='softmax')
        out = out(hidden_layers[-1])
        self.model = Model(input=sentence_input, output=out)

def train_and_test_model(nn, X_train, y_train, X_test, y_test,
                         batch_size, nb_epoch,
                         lr, beta_1, beta_2, epsilon):
    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    if nn.nb_labels == 2:
        nn.model.compile(loss='binary_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
    else:
        nn.model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
    nn.model.fit(X_train, y_train,
                 batch_size=batch_size, nb_epoch=nb_epoch,
                 validation_split=0.1)
    score, acc = nn.model.evaluate(X_test, y_test, batch_size=64)
    return nn, acc
