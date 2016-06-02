import sys
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from preprocessing.data_utils import load_data, load_dataset
from paths import SENTENCES_CSV

def train_test_bow(ngram_order):
    label_sets = ['full', 'function', '3way', 'in_out', 'man_nat']
    for label_set in label_sets:
        df = load_data(SENTENCES_CSV, labels=label_set)
        X, y, word2idx, l_enc = load_dataset(df, ngram_order=ngram_order)
        print "X shape: %s" % (X.shape,)
        print "y shape: %s" % (y.shape,)
        clf = SGDClassifier(loss='log',
                            alpha=0.1,
                            l1_ratio=0,
                            random_state=0)
        scores = cross_val_score(clf, X, y, cv=5, verbose=1)
        print '%s label mean cv accuracy: %.2f\n' % (label_set, np.mean(scores))

def train_test_feats():
    pass

def train_and_test_maxent(ngram_order=1, feats='bow'):
    print "ngram_order:", ngram_order
    print "features:", feats, '\n'
    if feats == 'bow':
        train_test_bow(ngram_order)
    elif feats == 'feats':
        train_test_feats()
