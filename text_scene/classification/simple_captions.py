import numpy as np
import pandas as pd
import scipy.misc
import skimage.io
import json
import pickle
import h5py
import glob
import argparse
import os
import sys

from itertools import islice, izip

from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Flatten, merge, RepeatVector, Lambda
from keras.layers import Dense, GRU, LSTM, Dropout, Activation, TimeDistributed
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Merge
from keras.utils.np_utils import to_categorical
from keras.utils.layer_utils import print_summary
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras import backend as K

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.data_utils import (
    load_dataset, sentences_df, load_bin_vec, add_unknown_words
)


def build_model(vocab_size, max_caption_len, weights_path, embedding_weights,
                scene_model=True):
    image_model = Sequential()
    image_model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    image_model.add(Convolution2D(64, 3, 3, activation='relu'))
    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(64, 3, 3, activation='relu'))
    image_model.add(MaxPooling2D((2,2), strides=(2,2)))

    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(128, 3, 3, activation='relu'))
    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(128, 3, 3, activation='relu'))
    image_model.add(MaxPooling2D((2,2), strides=(2,2)))

    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(256, 3, 3, activation='relu'))
    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(256, 3, 3, activation='relu'))
    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(256, 3, 3, activation='relu'))
    image_model.add(MaxPooling2D((2,2), strides=(2,2)))

    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(512, 3, 3, activation='relu'))
    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(512, 3, 3, activation='relu'))
    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(512, 3, 3, activation='relu'))
    image_model.add(MaxPooling2D((2,2), strides=(2,2)))

    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(512, 3, 3, activation='relu'))
    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(512, 3, 3, activation='relu'))
    image_model.add(ZeroPadding2D((1,1)))
    image_model.add(Convolution2D(512, 3, 3, activation='relu'))
    image_model.add(MaxPooling2D((2,2), strides=(2,2)))

    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(image_model.layers):
            # we don't look at the last (fully-connected) layers in
            # the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        image_model.layers[k].set_weights(weights)
    f.close()

    image_model.add(Flatten())
    image_model.add(Dense(128, activation='relu'))

    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 300,
                                 input_length=max_caption_len,
                                 mask_zero=True,
                                 weights=[embedding_weights]))
    language_model.add(GRU(output_dim=128, return_sequences=True,
                           dropout_W=0.5, dropout_U=0.5,
                           activation='relu'))
    language_model.add(TimeDistributed(Dense(128, activation='relu')))

    image_model.add(RepeatVector(max_caption_len))

    if scene_model:
        label_model = Sequential()
        label_model.add(Embedding(13, 100,
                                  input_length=1,
                                  mask_zero=False))
        label_model.add(Flatten())
        label_model.add(RepeatVector(max_caption_len))

    model = Sequential()
    if scene_model:
        model.add(Merge([image_model, language_model, label_model],
                        mode='concat', concat_axis=-1))
    else:
        model.add(Merge([image_model, language_model],
                        mode='concat', concat_axis=-1))
    model.add(GRU(256, return_sequences=False,
                  dropout_W=0.5, dropout_U=0.5,
                  activation='relu'))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model

def sample(preds, temperature=1.):
    # helper function to sample idx from probas array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def preprocess_im(im):
    im = scipy.misc.imresize(im, (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    return im

def stream(X):
    while True:
        for x_i in X:
            yield x_i

def imstream(imlookup, imfiles):
    while True:
        for imfile in imfiles:
            yield imlookup[imfile]

def nextwordstream(next_words):
    while True:
        for next_word in next_words:
            yield next_word

def minibatches(imstream, capstream, nextstream, scenestream,
                word2idx, nb_samples, scenes=False, batch_size=64):
    nb_batches = nb_samples // batch_size
    while True:
        for batch in range(nb_batches):
            ims = np.asarray(list(islice(imstream, batch_size)),
                             dtype=np.float32)
            captions = np.asarray(list(islice(capstream, batch_size)),
                                  dtype=np.float32)
            scenes = np.asarray(list(islice(scenestream, batch_size)),
                                dtype=np.int32)
            if scenes:
                X = [ims, captions, scenes]
            else:
                X = [ims, captions]
            y = np.asarray(list(islice(nextstream, batch_size)))
            y = to_categorical(y, nb_classes=len(word2idx)+1).astype('int32')
            yield X, y

def setup_generation(word_vecs, model_weights, scene_model=False):
    with open('../../models/l_enc.pkl') as p:
        l_enc = pickle.load(p)
    with open('../../models/word2idx.json') as j:
        word2idx = json.load(j)
    vocab_size = len(word2idx) + 1
    max_caption_len = 79
    word_vectors = load_bin_vec(word_vecs, word2idx)
    add_unknown_words(word_vectors, word2idx)
    embedding_weights = np.zeros((vocab_size,300))
    for word, index in word2idx.items():
        embedding_weights[index:,] = word_vectors[word]
    model = build_model(vocab_size, max_caption_len, '../../models/vgg16_weights.h5',
                        embedding_weights, scene_model=scene_model)
    model.load_weights(model_weights)
    return model, word2idx, l_enc

def generate(model, imfile, word2idx, l_enc, scene=None, temperature=1.):
    idx2word = {i:w for w,i in word2idx.items()}
    im = preprocess_im(scipy.misc.imread(imfile))
    im = np.expand_dims(im, axis=0)
    start_token, end_token = word2idx['<s>'], word2idx['<e>']
    input_caption = [start_token]
    input_caption_x = np.expand_dims(np.asarray(input_caption), axis=0)
    input_caption_x = pad_sequences(input_caption_x, padding='post', maxlen=79)
    if scene:
        label = l_enc.transform([scene])
        x = [im, input_caption_x, label]
    else:
        x = [im, input_caption_x]

    generated = ""

    preds = model.predict(x)[0]
    next_idx = sample(preds, temperature)
    next_word = idx2word[next_idx]
    generated += " " + next_word
    sys.stdout.write(next_word)

    while next_idx != end_token:
        if scene:
            x = [im, input_caption_x, label]
        else:
            x = [im, input_caption_x]

        preds = model.predict(x, batch_size=1, verbose=0)[0]
        next_idx = sample(preds, temperature)
        input_caption.append(next_idx)
        input_caption_x = np.expand_dims(np.asarray(input_caption), axis=0)
        input_caption_x = pad_sequences(input_caption_x, padding='post',
                                        maxlen=79)

        next_word = idx2word[next_idx]
        generated += " " + next_word
        sys.stdout.write(' ' + next_word)
        sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int)
    parser.add_argument('--vggweights', type=str)
    parser.add_argument('--modelweights', type=str)
    parser.add_argument('--wordvecs', type=str)
    parser.add_argument('--images', type=str)
    parser.add_argument('--scenes', action='store_true')
    args = parser.parse_args()

    vggweights_path = args.vggweights
    modelweights_path = args.modelweights
    word_vecs = args.wordvecs
    img_directory = args.images
    emb_dim = args.dim
    scene_model = args.scenes

    df = sentences_df(keep_filename=True, special_tokens=True, drop_unk=True)
    imlookup = {}
    for imfile in set(df.img_file):
        impath = os.path.join(img_directory, imfile)
        im = scipy.misc.imread(impath)
        im = preprocess_im(im)
        imlookup[imfile] = im

    # X sentences should have the start token but not the end token,
    # since they'll be the input to the RNN.
    # Y sentences should have the end token, but not the start token,
    # since they'll be the predictions given by the RNN.
    print "Loading captions..."
    X, scene_labels, word2idx, l_enc = load_dataset(df, pad=True)
    with open('../../models/word2idx.json') as j:
        word2idx = json.load(j)
    with open('../../models/l_enc.pkl') as p:
        l_enc = pickle.load(p)

    # partial captions are word 0..n
    # next word is word n+1
    # create samples for n=1...N-1
    partial_captions = []
    next_words = []
    imfiles = []
    scenes = []
    for i, x_i in enumerate(X):
        indices = x_i[np.nonzero(x_i)[0]]
        for j in range(len(indices)-1):
            imfiles.append(df.img_file.iloc[i])
            scenes.append(np.asarray(l_enc.transform(df.label.iloc[i])))
            partial_captions.append(indices[:j+1])
            next_words.append(indices[j+1])
    partial_captions = pad_sequences(partial_captions, padding='post')

    print "pc shape", partial_captions.shape
    print "imfiles shape", len(imfiles)
    print "scenes shape", len(scenes)
    print "next_words shape", len(next_words)

    vocab_size = len(word2idx) + 1
    max_caption_len = partial_captions.shape[1]

    word_vectors = load_bin_vec(word_vecs, word2idx)
    add_unknown_words(word_vectors, word2idx)
    embedding_weights = np.zeros((vocab_size, emb_dim))
    for word, index in word2idx.items():
        embedding_weights[index,:] = word_vectors[word]

    model = build_model(vocab_size, max_caption_len, vggweights_path,
                        embedding_weights, scene_model=scene_model)
    model.load_weights(modelweights_path)
    print_summary(model.layers)

    partial_captions_train, partial_captions_test = (
        partial_captions[:280000],
        partial_captions[280000:])
    imfiles_train, imfiles_test = (
        imfiles[:280000],
        imfiles[280000])
    next_words_train, next_words_test = (
        next_words[:280000],
        next_words[280000:])
    scenes_train, scenes_test = (
        scenes[:280000],
        scenes[280000:])

    samples_per_epoch = partial_captions_train.shape[0]

    imstream = imstream(imlookup, imfiles_train)
    capstream = stream(partial_captions_train)
    nextstream = nextwordstream(next_words_train)
    scenestream = stream(scenes_train)
    generator = minibatches(imstream, capstream, nextstream, scenestream,
                            word2idx, samples_per_epoch, scenes=scene_model)

    if scene_model:
        checkpoint = ModelCheckpoint('../../models/weights.scenes.{epoch:02d}.h5')
    else:
        checkpoint = ModelCheckpoint('../../models/weights.{epoch:02d}.h5')
    model.fit_generator(generator,
                        nb_epoch=100,
                        samples_per_epoch=samples_per_epoch,
                        callbacks=[checkpoint])
