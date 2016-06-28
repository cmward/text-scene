import random
import sys
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.data_utils import sentences_df

"""Use an char-lstm to generate text. Train a model for each label.

Adapted from
github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
"""

df = sentences_df(labels='function')
labels = np.unique(df.columns)

df_rest = df[df.label == 'restaurant']
sents_rest = df_rest['sentence'].values
text = ' '.join(sent for sent in sents_rest)
print 'corpus length:', len(text)

chars = sorted(list(set(text)))
print 'total chars:', len(chars)
char_indices = dict((c,i) for i,c in enumerate(chars))
indices_char = dict((i,c) for i,c in enumerate(chars))

# cut text into sequences of maxlen chars
maxlen = 40
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

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print
    print '-' * 50
    print 'Iteration', iteration
    model.fit(X, y, batch_size=64, nb_epoch=1)

    if iteration % 5 != 0: continue
    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print
        print '----- diversity:', diversity 

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print '----- Generating with seed: "' + sentence + '"'
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
print