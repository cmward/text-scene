import numpy as np
import pandas as pd
import scipy.misc
import skimage.io
import h5py
import glob
import argparse
import os
import sys

from itertools import izip, islice

from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Flatten, merge, RepeatVector, Lambda
from keras.layers import Dense, GRU, LSTM, Dropout, Activation, TimeDistributed
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.utils.layer_utils import print_summary
from keras.engine.topology import InputSpec
from keras.callbacks import ModelCheckpoint
from keras import backend as K

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.data_utils import (
    load_dataset, sentences_df, load_bin_vec, add_unknown_words
)


class InitialStateGRU(GRU):

    # https://github.com/fchollet/keras/issues/2995
    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        if isinstance(x, (tuple, list)):
            x, custom_initial = x
        else:
            custom_initial = None
        input_shape = self.input_spec[0].shape

        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        elif custom_initial:
            initial_states = custom_initial
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)
        print "call input shape", input_shape
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

        def get_output_shape_for(self, input_shape):
            print "get output shape", input_shape
            return (input_shape[0], self.output_dim)


    def build(self, input_shape):
        print "Input shape:", input_shape
        self.input_spec = [InputSpec(shape=input_shape[0]),
                           InputSpec(shape=input_shape[1])]
        self.input_dim = input_shape[0][2] + input_shape[1][2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.consume_less == 'gpu':

            self.W = self.init((self.input_dim, 3 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 3 * self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))

            self.trainable_weights = [self.W, self.U, self.b]
        else:

            self.W_z = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_z'.format(self.name))
            self.U_z = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_z'.format(self.name))
            self.b_z = K.zeros((self.output_dim,), name='{}_b_z'.format(self.name))

            self.W_r = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_r'.format(self.name))
            self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_r'.format(self.name))
            self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))

            self.W_h = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_h'.format(self.name))
            self.U_h = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_h'.format(self.name))
            self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))

            self.trainable_weights = [self.W_z, self.U_z, self.b_z,
                                      self.W_r, self.U_r, self.b_r,
                                      self.W_h, self.U_h, self.b_h]

            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

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
    encode = Dense(emb_dim, activation='relu')(flatten)
    encode = RepeatVector(max_caption_len)(encode)
    #encode = Lambda(lambda x: K.expand_dims(x, dim=1),
    #                output_shape=(max_caption_len, emb_dim))(encode)

    # RNN to produce output sequence
    caption = Input(shape=(max_caption_len,), dtype='int32')
    embeddings = Embedding(input_dim=vocab_size,
                           output_dim=emb_dim,
                           mask_zero=True,
                           weights=[embedding_weights])
    embeddings = embeddings(caption)
    conditioned_embeddings = merge([encode, embeddings],
                                   mode='concat',
                                   concat_axis=2)
    # set initial hidden state to image encoding
    rnn = GRU(emb_dim, return_sequences=True, activation='relu')
    rnn = rnn(conditioned_embeddings)
    probas = TimeDistributed(Dense(vocab_size, activation='softmax'))(rnn)

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

def imstream(ic):
    # stream each image 5 times
    for im in ic:
        for _ in range(5):
            yield preprocess_im(im)

def captionstream(X):
    for caption in X:
        yield caption

# this is wrong
def one_hot_captionstream(captions, word2idx):
    for caption in captions:
        yield to_categorical(caption, nb_classes=len(word2idx)+1)

def minibatches(imstream, captionstream, oh_captionstream, batch_size=32,
                nb_samples=23285):
    nb_batches = nb_samples // batch_size
    for batch in range(nb_batches):
        ims = np.asarray(list(islice(imstream, batch_size)))
        captions = np.asarray(list(islice(captionstream, batch_size)))
        X = [ims, captions]
        y = np.asarray(list(islice(oh_captionstream, batch_size)))
        yield X, y

def sample():
    pass

def generate_caption(image_file):
    pass

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
    # load images and captions
    ic = skimage.io.imread_collection(os.path.join(img_directory, '*.jpg'),
                                      conserve_memory=True)
    df = sentences_df(keep_filename=True, special_tokens=True, drop_unk=True)

    # sort df by ImageCollection ordering
    img_order = [f.split('/')[-1] for f in ic.files]
    df['img_file_ord'] = pd.Categorical(
        df['img_file'],
        categories=img_order,
        ordered=True)
    df = df.sort_values(by='img_file_ord')

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

    ims_train, ims_test = ic[:4000], ic[4000:]
    X_sents_train, y_sents_train = X_sents[:20000], y_sents[:20000]
    X_sents_test, y_sents_test = X_sents[20000:], y_sents[20000:]


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

    samples_per_epoch = X_sents_train.shape[0]
    checkpoint = ModelCheckpoint('../../weights.{epoch:02d}.h5')
    for epoch in range(50):
        ims = imstream(ims_train)
        caps = captionstream(X_sents_train)
        oh_caps = one_hot_captionstream(y_sents_train, word2idx)
        batches = minibatches(ims, caps, oh_caps)
        model.fit_generator(minibatches(ims, caps, oh_caps),
                            samples_per_epoch=samples_per_epoch,
                            nb_epoch=1,
                            callbacks=[checkpoint])
        """j
        for X, y in batches:
            loss, acc = model.train_on_batch(X, y)
            sys.stdout.write('\rEpoch: %i --  Loss: %.4f -- Acc: %.4f' % (
                epoch, loss, acc))
            sys.stdout.flush()
        model.save_weights('../weights_epoch{}.h5'.format(epoch))
        print
        """

    test_ims = imstream(ims_test)
    test_caps = captionstream(X_sents_test)
    test_oh_caps = one_hot_captionstream(y_sents_test, word2idx)

    model.evaluate_generator(minibatches(test_ims, test_caps, test_oh_caps))

