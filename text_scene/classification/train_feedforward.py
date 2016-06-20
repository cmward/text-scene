import os
import sys
import time
import numpy as np
from collections import Counter
from keras.utils.np_utils import to_categorical, probas_to_classes
from keras.utils.layer_utils import print_summary
from sklearn.cross_validation import StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.data_utils import (
    load_bin_vec,
    sentences_df,
    load_dataset,
    add_unknown_words
)
from corpus_stats.frequency import print_label_frequencies
from feedforward import FeedforwardNN, train_and_test_model
from paths import SENTENCES_CSV


# hyperparameters
emb_dim = 300
batch_size = 64
nb_epoch = 15
lr = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08

def train(label_set='full', pool_mode='max', drop_unk=False, word_vecs=None):
    print "Loading data..."
    df = sentences_df(SENTENCES_CSV, labels=label_set, drop_unk=drop_unk)
    X, y, word2idx, l_enc = load_dataset(df, pad=True)
    print "X shape:", X.shape
    y_orig = y
    y_binary = to_categorical(y)
    labels = np.unique(y_orig)
    nb_labels = labels.shape[0]
    if drop_unk:
        label_set_str = label_set + ' (-unk)'
    else:
        label_set_str = label_set
    print "Number of labels: %i [%s]" % (nb_labels, label_set_str)
    if nb_labels > 2:
        y = y_binary
    maxlen = X.shape[1]
    vocab_size = len(word2idx) + 1 # 0 masking
    word_vectors = load_bin_vec(word_vecs, word2idx)
    add_unknown_words(word_vectors, word2idx)
    embedding_weights = np.zeros((vocab_size+1, emb_dim))
    for word, index in word2idx.items():
        embedding_weights[index,:] = word_vectors[word]
    print "Data loaded."

    params = [('batch_size',batch_size), ('nb_epoch',nb_epoch),
              ('lr',lr), ('beta_1',beta_1), ('beta_2',beta_2),
              ('epsilon',epsilon)]
    for (name, value) in params:
        print name + ':', value

    skf = StratifiedKFold(y_orig, n_folds=5, shuffle=True, random_state=0)
    cv_scores = []
    for i, (train, test) in enumerate(skf):
        start_time = time.time()
        nn = None
        nn = FeedforwardNN(vocab_size,
                           nb_labels,
                           emb_dim,
                           maxlen,
                           embedding_weights,
                           pool_mode=pool_mode)
        if i == 0:
            print_summary(nn.model.layers)
        nn, acc = train_and_test_model(nn, X[train], y[train], X[test], y[test],
                                   batch_size, nb_epoch,
                                   lr, beta_1, beta_2, epsilon)
        cv_scores.append(acc)
        train_time = time.time() - start_time
        print('\nLabel frequencies in y[test]')
        print_label_frequencies((y_orig[test], l_enc))
        y_pred = nn.model.predict(X[test])
        y_pred = probas_to_classes(y_pred)
        c = Counter(y_pred)
        total = float(len(y_pred))
        print('\nLabel frequencies in predict(y[test])')
        for label, count in c.most_common():
            print l_enc.inverse_transform(label), count, count / total
        print "\nfold %i/5 - time: %.2f s - acc: %.2f on %i samples" % \
            (i+1, train_time, acc, len(test))
    print "Avg cv accuracy: %.2f" % np.mean(cv_scores)
