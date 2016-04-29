import os
import sys
import numpy as np
from theano import tensor as T
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Lambda, Merge, merge
from keras.layers import Embedding, Input
from keras.layers import Convolution1D, MaxPooling1D
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras import backend as K


def max_1d(X):
    return K.max(X, axis=1)

def kmax_1d_numpy(X, k):
    idxs = np.argsort(X, axis=1)[:,X.shape[1]-k:]
    dim_0 = np.repeat(np.arange(idxs.shape[0]), idxs.shape[1]*idxs.shape[2])
    dim_1 = idxs.transpose(0,2,1).ravel()
    dim_2 = np.tile(np.repeat(np.arange(idxs.shape[2]), idxs.shape[1]),
                    idxs.shape[0])
    kmax = X[dim_0, dim_1, dim_2]
    return kmax.reshape((idxs.shape[0],idxs.shape[2],idxs.shape[1])).swapaxes(1,2)

def kmax_1d(X, k):
    """
    Max pooling operation that selects the k largest values
    along the timesteps axis.
    Arguments:
        X: 3d tensor of shape (samples, steps, input_dim)
        k: number of activations to keep per sample
    Returns:
        3d tensor of shape (samples, downsampled_steps, input_dim)
    """
    idxs = T.argsort(X, axis=1)[:,X.shape[1]-k:]
    dim_0 = T.repeat(T.arange(idxs.shape[0]), idxs.shape[1]*idxs.shape[2])
    dim_0 = T.cast(dim_0, 'int32')
    dim_1 = idxs.transpose(0,2,1).ravel()
    dim_1 = T.cast(dim_1, 'int32')
    dim_2 = T.tile(T.repeat(T.arange(idxs.shape[2]), idxs.shape[1]), idxs.shape[0])
    dim_2 = T.cast(dim_2, 'int32')
    kmax = X[dim_0, dim_1, dim_2]
    return kmax.reshape((idxs.shape[0],idxs.shape[2],idxs.shape[1])).swapaxes(1,2)

class KmaxCNN(object):
    """
    A CNN for text classification.
    embedding --> conv --> k-max pool --> softmax
    """
    def __init__(self, vocab_size, nb_labels, emb_dim, maxlen,
                 embedding_weights, filter_hs, nb_filters, dropout_p,
                 maxnorm_val, trainable_embeddings, pretrained_embeddings):
        self.nb_labels = nb_labels
        self.k = 3
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
        conv_pools = []
        for filter_h in filter_hs:
            conv = Convolution1D(nb_filters, filter_h, border_mode='same',
                                 activation='relu')
            conved = conv(x)
            pool = Lambda(kmax_1d, arguments={'k':self.k},
                          output_shape=(self.k, nb_filters))
            pooled = pool(conved)
            flattened = Flatten()(pooled)
            conv_pools.append(flattened)
        merged = merge(conv_pools, mode='concat')
        dropout = Dropout(dropout_p[1])(merged)
        if nb_labels == 2:
            out = Dense(1, activation='sigmoid',
                        W_constraint=maxnorm(maxnorm_val))
            out = out(dropout)
        else:
            out = Dense(nb_labels, activation='softmax',
                        W_constraint=maxnorm(maxnorm_val))
            out = out(dropout)
        self.model = Model(input=sentence_input, output=out)

class ParallelCNN(object):
    """
    A CNN for text classification.
    embedding --> conv --> 1-max pool --> softmax
    """
    def __init__(self, vocab_size, nb_labels, emb_dim, maxlen,
                 embedding_weights, filter_hs, nb_filters, dropout_p,
                 maxnorm_val, trainable_embeddings, pretrained_embeddings):
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
        conv_pools = []
        for filter_h in filter_hs:
            conv = Convolution1D(nb_filters, filter_h, border_mode='same',
                                 activation='relu')
            conved = conv(x)
            pool = Lambda(max_1d, output_shape=(nb_filters,))
            pooled = pool(conved)
            conv_pools.append(pooled)
        merged = merge(conv_pools, mode='concat')
        dropout = Dropout(dropout_p[1])(merged)
        if nb_labels == 2:
            out = Dense(1, activation='sigmoid',
                        W_constraint=maxnorm(maxnorm_val))
            out = out(dropout)
        else:
            out = Dense(nb_labels, activation='softmax',
                        W_constraint=maxnorm(maxnorm_val))
            out = out(dropout)
        self.model = Model(input=sentence_input, output=out)

class DeepCNN(object):
    def __init__(self, vocab_size, nb_labels, emb_dim, maxlen,
                 embedding_weights):
        self.nb_labels = nb_labels
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size+1,
                            output_dim=300,
                            input_length=maxlen,
                            dropout=0.2,
                            weights=[embedding_weights]))
        model.add(Convolution1D(16, 3, border_mode='same', activation='relu'))
        #model.add(Convolution1D(16, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling1D(pool_length=2))
        #model.add(Convolution1D(16, 3, border_mode='same', activation='relu'))
        model.add(Convolution1D(16, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling1D(pool_length=2))
        model.add(Flatten())
        #model.add(Lambda(max_1d, output_shape=(16,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        if nb_labels == 2:
            model.add(Dense(1, activation='sigmoid',
                            W_constraint=maxnorm(3)))
        else:
            model.add(Dense(nb_labels, activation='softmax',
                            W_constraint=maxnorm(3)))
        self.model = model

def create_model(vocab_size, nb_labels, emb_dim, maxlen,
                 embedding_weights, filter_hs, nb_filters,
                 dropout_p, maxnorm_val, trainable_embeddings,
                 pretrained_embeddings, model_type='parallel'):
    """
    Create CNN with architecture specified by `model_type.`
    Returns a ParallelCNN or DeepCNN object, with attribute
    `model` which is the Keras model for the CNN.
    """
    if model_type == 'deep':
        cnn = DeepCNN(vocab_size, nb_labels, emb_dim, maxlen,
                      embedding_weights)
    elif model_type == 'parallel':
        cnn = ParallelCNN(vocab_size, nb_labels, emb_dim, maxlen,
                          embedding_weights, filter_hs, nb_filters,
                          dropout_p, maxnorm_val, trainable_embeddings,
                          pretrained_embeddings)
    elif model_type == 'kmax':
        cnn = KmaxCNN(vocab_size, nb_labels, emb_dim, maxlen,
                      embedding_weights, filter_hs, nb_filters,
                      dropout_p, maxnorm_val, trainable_embeddings,
                      pretrained_embeddings)
    return cnn
