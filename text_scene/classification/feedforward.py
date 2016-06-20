import os
import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Lambda, Merge, merge
from keras.layers import Embedding, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K

class FeedforwardNN(object):
    def __init__(self, vocab_size, nb_labels, emb_dim, maxlen,
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
        else:
            pool = Lambda(lambda x: K.mean(x, axis=1), output_shape=(emb_dim,))
        pool_out = Dropout(0.5)(pool(x))
        """
        prev_layer = pool_out
        for layer_size in layer_sizes:
            hidden_in = Dense(layer_size, activation='relu')
            hidden_out = Dropout(0.5)(hidden_in(prev_layer))
            prev_layer = hidden_out
        """
        hidden_1 = Dense(256, activation='relu')
        hidden_1_out = Dropout(0.5)(hidden_1(pool_out))
        hidden_2 = Dense(256, activation='relu')
        hidden_2_out = Dropout(0.5)(hidden_2(hidden_1_out))
        #hidden_3 = Dense(256, activation='relu')
        #hidden_3_out = Dropout(0.5)(hidden_3(hidden_2_out))
        if self.nb_labels == 2:
            out = Dense(nb_labels, activation='sigmoid')
        else:
            out = Dense(nb_labels, activation='softmax')
        out = out(hidden_2_out)
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