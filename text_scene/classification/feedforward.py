import os
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, TimeDistributed
from keras.layers import Flatten, Lambda, Merge, merge
from keras.layers import Embedding, Input, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K

class FeedforwardNN(object):
    def __init__(self, vocab_size, nb_labels, emb_dim, maxlen, layer_sizes,
                 embedding_weights, pool_mode='max', activation='prelu',
                 dropout_p=[0.7,0.5,0.5]):

        print "Activation:", activation
        print "Pool mode:", pool_mode
        print "Dropout", dropout_p

        self.nb_labels = nb_labels

        sentence_input = Input(shape=(maxlen,), dtype='int32')
        x = Embedding(input_dim=vocab_size+1,
                      output_dim=emb_dim,
                      input_length=maxlen,
                      weights=[embedding_weights])
        x = x(sentence_input)
        x = Dropout(dropout_p[0])(x)

        if pool_mode == 'sum':
            pool = Lambda(lambda x: K.sum(x, axis=1),
                          output_shape=(emb_dim,))(x)

        elif pool_mode == 'max':
            pool = Lambda(lambda x: K.max(x, axis=1),
                          output_shape=(emb_dim,))(x)

        elif pool_mode == 'mean':
            pool = Lambda(lambda x: K.mean(x, axis=1),
                          output_shape=(emb_dim,))(x)

        elif pool_mode == 'concat':
            pool = Flatten()(x)

        else:
            dense = TimeDistributed(Dense(300))(x)
            dense_bn = BatchNormalization()(dense)
            if activation == 'relu':
                dense_activation = Activation('relu')(dense_bn)
            elif activation == 'tanh':
                dense_activation = Activation('tanh')(dense_bn)
            elif activation == 'prelu':
                dense_activation = PReLU()(dense_bn)
            elif activation == 'leakyrelu':
                dense_activation = LeakyReLU()(dense_bn)
            else: #ELU
                dense_activation = ELU()(dense_bn)
            pool = Lambda(lambda x: K.sum(x, axis=1),
                          output_shape=(300,))(dense_activation)

        pool_out = Dropout(dropout_p[1])(pool)
        hidden_layers = []
        prev_layer = pool_out

        for layer_size in layer_sizes:
            hidden_in = Dense(layer_size)(prev_layer)
            hidden_bn = BatchNormalization()(hidden_in)
            if activation == 'relu':
                hidden_activation = Activation('relu')(hidden_bn)
            elif activation == 'tanh':
                hidden_activation = Activation('tanh')(hidden_bn)
            elif activation == 'prelu':
                hidden_activation = PReLU()(hidden_bn)
            elif activation == 'leakyrelu':
                hidden_activation = LeakyReLU()(hidden_bn)
            else: #ELU
                hidden_activation = ELU()(hidden_bn)
            hidden_out = Dropout(dropout_p[2])(hidden_activation)
            hidden_layers.append(hidden_out)
            prev_layer = hidden_out

        if self.nb_labels == 2:
            out = Dense(1, activation='sigmoid')
        else:
            out = Dense(nb_labels, activation='softmax')
        if not hidden_layers:
            out = out(pool_out)
        else:
            out = out(hidden_layers[-1])
        self.model = Model(input=sentence_input, output=out)

class FastText(object):
    def __init__(self, vocab_size, nb_labels, emb_dim, maxlen, layer_sizes,
                 embedding_weights, pool_mode='max', activation='relu'):
        self.nb_labels = nb_labels
        sentence_input = Input(shape=(maxlen,), dtype='int32')
        x = Embedding(input_dim=vocab_size+1,
                      output_dim=emb_dim,
                      input_length=maxlen,
                      weights=None)
        x = x(sentence_input)
        pool = Lambda(lambda x: K.mean(x, axis=1), output_shape=(emb_dim,))
        pool_out = pool(x)
        if self.nb_labels == 2:
            out = Dense(1, activation='sigmoid')
        else:
            out = Dense(nb_labels, activation='softmax')
        out = out(pool_out)
        self.model = Model(input=sentence_input, output=out)

def train_and_test_model(nn, X_train, y_train, X_test, y_test,
                         batch_size, nb_epoch,
                         lr, beta_1, beta_2, epsilon,
                         val_split):
    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    if nn.nb_labels == 2:
        nn.model.compile(loss='binary_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
    else:
        nn.model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
    nn.model.fit(X_train, y_train, nb_epoch=nb_epoch,
                 batch_size=batch_size,
                 validation_split=val_split)
    score, acc = nn.model.evaluate(X_test, y_test, batch_size=64)
    return nn, acc
