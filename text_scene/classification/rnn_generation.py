import random
import sys
import os
import random
import numpy as np
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers import LSTM, GRU, Embedding
from keras.utils.np_utils import to_categorical
from keras.utils.layer_utils import print_summary
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.data_utils import (
    sentences_df,
    load_dataset,
    load_bin_vec,
    add_unknown_words
)
from paths import GENERATED_TEXT

def log(content, outfile):
    """Write to stdout and outfile. outfile is open file"""
    outfile.write(content)
    sys.stdout.write(content)

# load sentences
df = sentences_df(labels='function')
labels = np.unique(df.label)
text = '\n'.join(s for s in df.sentence.values).split()

sents, _, word2idx, l_enc = load_dataset(df, pad=True)
idx2word = {i: w for w,i in word2idx.items()}
maxlen = sents.shape[1]
vocab = word2idx.keys()

# cut X into sequences of 10 words
# given 9 words, predict 10th
seqlen = 9
step = 2
sentences = []
next_words = []
mask = np.nonzero(sents)
text = sents[mask].astype(np.int32)
for i in range(0, len(text) - seqlen, step):
    sentences.append(text[i: i + seqlen])
    next_words.append(text[i + seqlen])
print 'nb sequences:', len(sentences)

print 'Vectorization...'
X = np.zeros((len(sentences), seqlen), dtype=np.int32)
y = np.zeros((len(sentences), len(vocab)+1), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t] = word
    y[i, word] = 1


# load word vectors
word_vecs = sys.argv[1]
word_vectors = load_bin_vec(word_vecs, word2idx)
add_unknown_words(word_vectors, word2idx)
embedding_weights = np.zeros((len(vocab)+1, 300))
for word, index in word2idx.items():
    embedding_weights[index,:] = word_vectors[word]
print "X shape:", X.shape, "y shape:", y.shape
print "len(vocab):", len(vocab)

# build model
sentence_input = Input(shape=(seqlen,), dtype='int32')
x = Embedding(input_dim=len(vocab)+1,
              output_dim=300,
              input_length=seqlen,
              weights=[embedding_weights],
              dropout=0.2)
x = x(sentence_input)
gru_1 = GRU(512, return_sequences=True, activation='relu')
gru_1_out = Dropout(0.5)(gru_1(x))
gru_2 = GRU(512, return_sequences=False, activation='relu')
gru_2_out = Dropout(0.5)(gru_2(gru_1_out))
softmax = Dense(len(vocab)+1, activation='softmax')(gru_2_out)
model = Model(input=sentence_input, output=softmax)

model.compile(loss='categorical_crossentropy', optimizer='adam')
print_summary(model.layers)

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

out = open('datafiles/word_generated.txt', 'a')

for iteration in range(1, 61):
    model.fit(X, y, batch_size=64, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        log('----- diversity: %f' % diversity, out)

        generated = ''
        sentence = text[start_index: start_index + seqlen]
        generated += ' '.join([idx2word[i] for i in sentence])

        log('----- Generating with seed: "' + generated + '"\n', out)
        log(generated, out)

        X_pred = np.zeros((1, seqlen), dtype=np.int32)
        for i in range(100):
            for t, word in enumerate(sentence):
                X_pred[0, t] = word

            preds = model.predict(X_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = idx2word[next_index]

            generated += next_word
            sentence = np.append(sentence[1:], [next_index])
            print sentence

            log(next_word + ' ', out)
            sys.stdout.flush()
        log('\n', out)

