import sys
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import cross_val_score
from paths import SENTENCES_CSV
from preprocessing.data_utils import (
    load_data,
    load_dataset, 
    print_label_frequencies
)

def train_and_test_nb(ngram_order=1):
    print "ngram order:", ngram_order
    label_sets = ['full', 'function', '3way', 'in_out', 'man_nat']
    for label_set in label_sets:
        df = load_data(SENTENCES_CSV, labels=label_set)
        X, y, word2idx, l_enc = load_dataset(df, ngram_order=ngram_order)
        print "X shape: %s" % (X.shape,)
        print "y shape: %s" % (y.shape,)
        clf = BernoulliNB()
        scores = cross_val_score(clf, X, y, cv=5)
        print_label_frequencies(df)
        print '%s label mean cv accuracy: %.2f\n' % (label_set, np.mean(scores))

