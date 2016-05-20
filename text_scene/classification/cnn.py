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
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras import backend as K

#TODO refactor parallel, kmax, and col into one class
#TODO parameterize deep
#TODO CNN superclass

def int32(X):
    return T.cast(X, 'int32')

def transpose(X):
    return K.transpose(x)

def max_1d(X):
    return K.max(X, axis=1)

def expand_channel_dim(X):
    return K.permute_dimensions(X, (0, 'x', 1, 2))

def transpose_in_batch(X):
    """
    Transpose the first and second axes,
    keeping the batch dimension the same.
    """
    return K.permute_dimensions(X, (0, 2, 1))

def vstack_in_batch(x):
    """
    `x` is a list of tensors.
    vstack that acts like np.vstack, i.e., if a and b have shape
    (1,5,20), return an array of shape (2,5,20)
    """
    return K.reshape(T.stack(x), (2, 200, 16))

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
    idxs = int32(T.argsort(X, axis=1)[:,X.shape[1]-k:])
    dim_0 = int32(T.repeat(T.arange(idxs.shape[0]), idxs.shape[1]*idxs.shape[2]))
    dim_1 = int32(idxs.transpose(0,2,1).ravel())
    dim_2 = int32(T.tile(T.repeat(T.arange(idxs.shape[2]), idxs.shape[1]),
                         idxs.shape[0]))
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
        self.k = 4
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
        dropped = Dropout(dropout_p[1])(merged)
        if nb_labels == 2:
            out = Dense(1, activation='sigmoid',
                        W_constraint=maxnorm(maxnorm_val))
            out = out(dropped)
        else:
            out = Dense(nb_labels, activation='softmax',
                        W_constraint=maxnorm(maxnorm_val))
            out = out(dropped)
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
                 embedding_weights, filter_hs, nb_filters,
                 dropout_p, trainable_embeddings,
                 pretrained_embeddings):
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
        expand = Lambda(expand_channel_dim, output_shape=(1, maxlen, emb_dim))
        x_expanded = expand(x)
        # CONV -> BATCHNORM -> RELU
        conv1 = Convolution2D(2, 3, 3, border_mode='same')(x_expanded)
        bn1 = BatchNormalization()(conv1)
        relu1 = Activation('relu')(bn1)
        # CONV -> BATCHNORM -> RELU -> POOL
        conv2 = Convolution2D(2, 3, 3, border_mode='same')(relu1)
        bn2 = BatchNormalization()(conv2)
        relu2 = Activation('relu')(bn2)
        pool1 = MaxPooling2D(pool_size=(2,2))(relu2)
        # CONV -> BATCHNORM -> RELU
        conv3 = Convolution2D(2, 3, 3, border_mode='same')(pool1)
        bn3 = BatchNormalization()(conv3)
        relu3 = Activation('relu')(bn3)
        # CONV -> BATCHNORM -> RELU -> POOL
        conv4 = Convolution2D(2, 3, 3, border_mode='same')(relu3)
        bn3 = BatchNormalization()(conv4)
        relu4 = Activation('relu')(bn3)
        pool2 = MaxPooling2D(pool_size=(2,2))(relu4)
        # FC -> BATCHNORM -> SOFTMAX
        flat = Flatten()(pool2)
        drop1 = Dropout(0.5)(flat)
        fc1 = Dense(64)(drop1)
        bn4 = BatchNormalization()(fc1)
        relu5 = Activation('relu')(bn4)
        drop2 = Dropout(0.5)(relu5)
        if nb_labels == 2:
            fc2 = Dense(1, activation='sigmoid', W_regularizer=l2(0.01))
            out = fc2(drop2)
        else:
            fc2 = Dense(nb_labels, activation='softmax', W_regularizer=l2(0.01))
            out = fc2(drop2)
        self.model = Model(input=sentence_input, output=out)

class ParallelColumnCNN(object):
    """
    Same as ParallelCNN except that feature maps are of size
    (maxlen,1), i.e., instead of convolving over ngrams, we
    convolve over word embedding feature columns.
    """
    # (79,10) filters, add parallel conv layer k = ~100
    def __init__(self, vocab_size, nb_labels, emb_dim, maxlen,
                 embedding_weights, filter_hs, nb_filters, dropout_p,
                 maxnorm_val, trainable_embeddings, pretrained_embeddings):
        self.nb_labels = nb_labels
        self.k = 100
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
        # transpose sentence matrix to use convolution1d to convolve
        # over feature rows rather than word rows
        x_T = Lambda(transpose_in_batch, output_shape=(emb_dim, maxlen))(x)
        conv_pools = []
        for filter_h in filter_hs:
            conv = Convolution1D(nb_filters, filter_h,
                                 border_mode='same', activation='relu')
            conved = conv(x_T)
            pool = Lambda(kmax_1d, arguments={'k':self.k},
                          output_shape=(self.k,nb_filters))
            #pool = Lambda(max_1d, output_shape=(nb_filters,))
            pooled = pool(conved)
            #flat = Flatten()(pooled)
            conv_pools.append(pooled)
            #conv_pools.append(flat)
        #merged = merge(conv_pools, mode='concat')
        merged = Lambda(vstack_in_batch,
                        output_shape=(len(filter_hs)*self.k,nb_filters))(conv_pools)
        conv_pools2 = []
        for filter_h in filter_hs:
            conv = Convolution1D(nb_filters, filter_h,
                                 border_mode='same', activation='relu')
            conved = conv(merged)
            pool = Lambda(max_1d, output_shape=(nb_filters,))
            pooled = pool(conved)
            #flat = Flatten()(pooled)
            conv_pools2.append(pooled)
            #conv_pools.append(flat)
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
                      embedding_weights, filter_hs, nb_filters,
                      dropout_p, trainable_embeddings,
                      pretrained_embeddings)
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
    elif model_type == 'col':
        cnn = ParallelColumnCNN(vocab_size, nb_labels, emb_dim, maxlen,
                                embedding_weights, filter_hs, nb_filters,
                                dropout_p, maxnorm_val, trainable_embeddings,
                                pretrained_embeddings)
    else:
        RuntimeError('Must enter a valid model type.')
    return cnn

def train_and_test_model(cnn, X_train, y_train, X_test, y_test,
                         batch_size, nb_epoch,
                         lr, beta_1, beta_2, epsilon):
    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    if cnn.nb_labels == 2:
        cnn.model.compile(loss='binary_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
    else:
        cnn.model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
    cnn.model.fit(X_train, y_train,
                  batch_size=batch_size, nb_epoch=nb_epoch,
                  validation_split=0.2)
    score, acc = cnn.model.evaluate(X_test, y_test, batch_size=64)
    return acc
