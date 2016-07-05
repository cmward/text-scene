import os
import sys
import numpy as np
from theano import tensor as T
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Lambda, Merge, merge
from keras.layers import Embedding, Input, Permute
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras import backend as K
from cnn import max_1d

class EnsembleNN(object):
    def __init__(self, vocab_size, nb_labels, emb_dim, maxlen,
                 embedding_weights, filter_hs, nb_filters, dropout_p,
                 trainable_embeddings, pretrained_embeddings, layer_sizes,
                 pool_mode='sum'):

        self.nb_labels = nb_labels
        if pretrained_embeddings is False:
            embedding_weights = None
        else:
            embedding_weights = [embedding_weights]

        sentence_input = Input(shape=(maxlen,), dtype='int32')
        x = Embedding(input_dim=vocab_size+1, output_dim=emb_dim,
                      input_length=maxlen, dropout=dropout_p[0],
                      weights=embedding_weights,
                      trainable=trainable_embeddings)
        x = x(sentence_input)

        # convnet
        conv_pools = []
        for filter_h in filter_hs:
            conv = Convolution1D(nb_filters, filter_h, border_mode='same',
                                 W_constraint=maxnorm(2))
            conved = conv(x)
            batchnorm = BatchNormalization()(conved)
            conved_relu = PReLU()(batchnorm)
            conv_pool = Lambda(max_1d, output_shape=(nb_filters,))
            conv_pooled = conv_pool(conved_relu)
            conv_pools.append(conv_pooled)
        conv_merged = merge(conv_pools, mode='concat')

        # feedforward
        if pool_mode == 'sum':
            ff_pool = Lambda(lambda x: K.sum(x, axis=1), output_shape=(emb_dim,))
        elif pool_mode == 'max':
            ff_pool = Lambda(lambda x: K.max(x, axis=1), output_shape=(emb_dim,))
        elif pool_mode == 'mean':
            ff_pool = Lambda(lambda x: K.mean(x, axis=1), output_shape=(emb_dim,))
        else: # concat
            ff_pool = Flatten()
        ff_pool_out = Dropout(0.5)(ff_pool(x))
        hidden_layers = []
        prev_layer = ff_pool_out
        for layer_size in layer_sizes:
            hidden_in = Dense(layer_size)(prev_layer)
            hidden_bn = BatchNormalization()(hidden_in)
            hidden_activation = PReLU()(hidden_bn)
            hidden_out = Dropout(0.5)(hidden_activation)
            hidden_layers.append(hidden_out)
            prev_layer = hidden_out

        # combine the two models
        combined = merge([hidden_layers[-1], conv_merged], mode='concat')
        combined_out = Dropout(0.5)(combined)
        if self.nb_labels == 2:
            out = Dense(1, activation='sigmoid')
        else:
            out = Dense(nb_labels, activation='softmax')
        out = out(combined_out)
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
    return acc
