import random
import sys
import os
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.utils.layer_utils import print_summary
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.data_utils import sentences_df
from paths import GENERATED_TEXT

"""Use an char-lstm to generate text. Train a model for each label.

Adapted from
github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
"""


def log(content, outfile):
    """Write to stdout and outfile. outfile is open file"""
    outfile.write(content)
    sys.stdout.write(content)

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def train_and_generate(nb_chars, X, y, model, text, maxlen, chars,
                       indices_char, char_indices, out):
    # train the model, output generated text after each iteration
    for iteration in range(1):
        log('-' * 50 + '\n', out)
        log('Iteration %i\n' % iteration, out)
        model.fit(X, y, batch_size=64, nb_epoch=1)
        if iteration % 10 == 0:
            generate(nb_chars, model, text, maxlen, chars,
                     indices_char, char_indices, out)
        log('\n', out)

def generate(nb_chars, model, text, maxlen, chars,
             indices_char, char_indices, out):

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        log('----- diversity: %f' % diversity, out)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence

        log('----- Generating with seed: "' + sentence + '"\n', out)
        log(generated, out)

        for i in range(nb_chars):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            log(next_char, out)
            sys.stdout.flush()
        log('\n', out)

def main():
    label_set = 'function'
    df = sentences_df(labels=label_set)
    labels = np.unique(df.label.values)

    out = open(GENERATED_TEXT, 'a')

    for label in labels:
        label_df = df[df.label == label]
        sents = label_df['sentence'].values
        text = '\n'.join(sent for sent in sents)
        print 'corpus length:', len(text)

        chars = sorted(list(set(text)))
        print 'total chars:', len(chars)
        char_indices = dict((c,i) for i,c in enumerate(chars))
        indices_char = dict((i,c) for i,c in enumerate(chars))

        # cut text into sequences of maxlen chars
        maxlen = 60
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print 'nb sequences:', len(sentences)

        print 'Vectorization...'
        X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        print 'Build model...'
        model = Sequential()
        model.add(GRU(256, return_sequences=True,
                      input_shape=(maxlen, len(chars)), activation='relu',))
        model.add(Dropout(0.5))
        model.add(GRU(256, return_sequences=False, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(chars), activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print_summary(model.layers)

        log('label: %s\n' % label, out)
        train_and_generate(600, X, y, model, text, maxlen, chars,
                           indices_char, char_indices, out)

    out.close()


if __name__ == '__main__':
    main()
