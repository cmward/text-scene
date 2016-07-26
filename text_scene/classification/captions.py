import numpy as np
import scipy.misc
import h5py
import glob
import argparse
import os
import sys

from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Flatten, merge, RepeatVector, Lambda
from keras.layers import Dense, GRU, Dropout, Activation, TimeDistributed
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.utils.layer_utils import print_summary
from keras import backend as K

from preprocessing.data_utils import load_dataset, sentences_df


def build_model(vocab_size, max_caption_len, emb_dim):
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
    encode = Dense(emb_dim, activation='relu')(flatten)
    encode = Lambda(lambda x: K.expand_dims(x, dim=1),
                    output_shape=(1, emb_dim))(encode)

    # RNN to produce output sequence
    partial_caption = Input(shape=(max_caption_len,), dtype='int32')
    embeddings = Embedding(input_dim=vocab_size,
                           output_dim=emb_dim)
    embeddings = embeddings(partial_caption)
    x = merge([encode, embeddings], mode='concat', concat_axis=1)
    gru = GRU(emb_dim, return_sequences=True, activation='relu')(x)
    probas = TimeDistributed(Dense(vocab_size, activation='softmax'))(gru)

    model = Model(input=[image, partial_caption], output=probas)

    model.compile(loss='categorical_crossentropy', optimizer='adam')

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

def load_images(img_directory, df):
    # read in and preprocess images
    images = np.zeros((df.shape[0]/5 + 1, 3, 224, 224), dtype=np.float32)
    imgs_to_load = df.img_file
    seen = set()
    nb_unique_imgs = len(set(df.img_file))

    i = 0
    for img_to_load in imgs_to_load:

        if img_to_load in seen:
            continue

        if i % 500 == 0:
            sys.stderr.write("\rLoading image %i of %i" % (i, nb_unique_imgs))
            sys.stderr.flush()

        seen.add(img_to_load)
        impath = os.path.join(img_directory, img_to_load)
        im = scipy.misc.imread(impath)
        im = scipy.misc.imresize(im, (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        images[i, ...] = im

        i += 1

    # repeat each image 5 times since they each have 5 captions
    images = np.repeat(images, 5)

    print "Images loaded"
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--images', type=str)
    args = parser.parse_args()

    weights_path = args.weights
    img_directory = args.images
    emb_dim = args.dim

    # add special start and end tokens

    # read in sentences
    df = sentences_df(keep_filename=True)
    images = load_images(img_directory, df)
    X, y, word2idx, l_enc = load_dataset(df, pad=True)
    vocab_size = len(word2idx) + 1
    max_caption_len = X.shape[1]
    sents = np.asarray([to_categorical(x_i, nb_classes=len(word2idx)+1)
                        for x_i in X],
                       dtype=np.float32)

    model = load_vgg_weights(build_model(vocab_size, X.shape[1], emb_dim),
                             weights_path)
    print "VGG 16 weights loaded."

    print_summary(model.layers)
