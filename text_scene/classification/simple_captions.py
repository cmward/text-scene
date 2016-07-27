import numpy as np
import pandas as pd
import scipy.misc
import skimage.io
import h5py
import glob
import argparse
import os
import sys

from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Flatten, merge, RepeatVector, Lambda
from keras.layers import Dense, GRU, LSTM, Dropout, Activation, TimeDistributed
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.utils.layer_utils import print_summary
from keras.callbacks import ModelCheckpoint
from keras import backend as K

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.data_utils import (
    load_dataset, sentences_df, load_bin_vec, add_unknown_words
)


def build_model(vocab_size, max_caption_len, emb_dim, embedding_weights):
    # build the VGG16 network
    image = Input(shape=(3, 224, 224))
    pad_1_1 = ZeroPadding2D((1,1),input_shape=(3,224,224))(image)
    conv_1_1 = Convolution2D(64, 3, 3, activation='relu')(pad_1_1)
    pad_1_2 = ZeroPadding2D((1,1))(conv_1_1)
    conv_1_2 = Convolution2D(64, 3, 3, activation='relu')(pad_1_2)
    pool_1 = MaxPooling2D((2,2), strides=(2,2))(conv_1_2)

    pad_2_1 = ZeroPadding2D((1,1))(pool_1)
    conv_2_1 = Convolution2D(128, 3, 3, activation='relu')(pad_2_1)
    pad_2_2 = ZeroPadding2D((1,1))(conv_2_1)
    conv_2_2 = Convolution2D(128, 3, 3, activation='relu')(pad_2_2)
    pool_2 = MaxPooling2D((2,2), strides=(2,2))(conv_2_2)

    pad_3_1 = ZeroPadding2D((1,1))(pool_2)
    conv_3_1 = Convolution2D(256, 3, 3, activation='relu')(pad_3_1)
    pad_3_2 = ZeroPadding2D((1,1))(conv_3_1)
    conv_3_2 = Convolution2D(256, 3, 3, activation='relu')(pad_3_2)
    pad_3_3 = ZeroPadding2D((1,1))(conv_3_2)
    conv_3_3 = Convolution2D(256, 3, 3, activation='relu')(pad_3_3)
    pool_3 = MaxPooling2D((2,2), strides=(2,2))(conv_3_3)

    pad_4_1 = ZeroPadding2D((1,1))(pool_3)
    conv_4_1 = Convolution2D(512, 3, 3, activation='relu')(pad_4_1)
    pad_4_2 = ZeroPadding2D((1,1))(conv_4_1)
    conv_4_2 = Convolution2D(512, 3, 3, activation='relu')(pad_4_2)
    pad_4_3 = ZeroPadding2D((1,1))(conv_4_2)
    conv_4_3 = Convolution2D(512, 3, 3, activation='relu')(pad_4_3)
    pool_4 = MaxPooling2D((2,2), strides=(2,2))(conv_4_3)

    pad_5_1 = ZeroPadding2D((1,1))(pool_4)
    conv_5_1 = Convolution2D(512, 3, 3, activation='relu')(pad_5_1)
    pad_5_2 = ZeroPadding2D((1,1))(conv_5_1)
    conv_5_2 = Convolution2D(512, 3, 3, activation='relu')(pad_5_2)
    pad_5_3 = ZeroPadding2D((1,1))(conv_5_2)
    conv_5_3 = Convolution2D(512, 3, 3, activation='relu')(pad_5_3)
    pool_5 = MaxPooling2D((2,2), strides=(2,2))(conv_5_3)

    # encode image into `emb_dim`-dimensional vector
    flatten = Flatten()(pool_5)
    image_encoder = Dense(128, activation='relu')(flatten)

    # RNN to produce output sequence
    caption = Input(shape=(max_caption_len,), dtype='int32')
    embeddings = Embedding(input_dim=vocab_size,
                           output_dim=emb_dim,
                           mask_zero=True,
                           weights=[embedding_weights])
    embeddings = embeddings(caption)
    caption_encoder = GRU(128, return_sequences=True, activation='relu')(embeddings)
    caption_encoder_out = TimeDistributed(Dense(128, activation='relu'))

    image_encoder = RepeatVector(max_caption_len)(image_encoder)
    encoding = merge([image_encoder, caption_encoder_out],
                     mode='concat',
                     concat_axis=-1)

    rnn = GRU(256, return_sequences=False, activation='relu')(encoding)
    probas = Dense(vocab_size, activation='softmax')

    model = Model(input=[image, caption], output=probas)

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model

def load_vgg_weights(model, weights_path):
    # load pretrained model weights
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers[1:len(model.layers)-8]):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k+1].set_weights(weights)
    f.close()
    return model

def preprocess_im(im):
    im = scipy.misc.imresize(im, (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    return im

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--wordvecs', type=str)
    parser.add_argument('--images', type=str)
    args = parser.parse_args()

    weights_path = args.weights
    word_vecs = args.wordvecs
    img_directory = args.images
    emb_dim = args.dim

    print "Loading images..."
    df = sentences_df(keep_filename=True, special_tokens=True, drop_unk=True)
    ims = np.zeros((df.shape[0], 3, 224, 224))
    for i, row in df.iterrows():
        impath = os.path.join(img_directory, row['img_file'])
        im = scipy.misc.imread(impath)
        im = preprocess_im(im)
        ims[i, ...] = im

    # X sentences should have the start token but not the end token,
    # since they'll be the input to the RNN.
    # Y sentences should have the end token, but not the start token,
    # since they'll be the predictions given by the RNN.
    print "Loading captions..."
    _, _, word2idx, _ = load_dataset(df, pad=True, truncate=True)
    Xdf = df.copy()
    ydf = df.copy()
    Xdf['sentence'] = Xdf['sentence'].apply(lambda x: ' '.join(x.split()[:-1]))
    ydf['sentence'] = ydf['sentence'].apply(lambda x: ' '.join(x.split()[1:]))
    X_sents, _, _, _ = load_dataset(Xdf, pad=True, word2idx=word2idx)
    y_sents, _, _, _ = load_dataset(ydf, pad=True, word2idx=word2idx)
    scene_labels = Xdf.label

    vocab_size = len(word2idx) + 1
    max_caption_len = X_sents.shape[1]

    ims_train, ims_test = ims[:4000], ims[4000:]
    X_sents_train, y_sents_train = X_sents[:20000], y_sents[:20000]
    X_sents_test, y_sents_test = X_sents[20000:], y_sents[20000:]

    X_train = [ims_train, X_sents_train]
    X_test = [ims_test, X_sents_test]

    word_vectors = load_bin_vec(word_vecs, word2idx)
    add_unknown_words(word_vectors, word2idx)
    embedding_weights = np.zeros((vocab_size, emb_dim))
    for word, index in word2idx.items():
        embedding_weights[index,:] = word_vectors[word]

    model = load_vgg_weights(build_model(vocab_size,
                                         max_caption_len,
                                         emb_dim,
                                         embedding_weights),
                             weights_path)
    print "VGG 16 weights loaded."

    print_summary(model.layers)

        

