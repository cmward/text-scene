import sys
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from preprocessing.data_utils import load_data, load_dataset
from paths import SENTENCES_CSV


"""
Usage:
    python maxent.py <mode> <ngram_order>
        mode := --feats | --bow
        ngram_order := int
"""

def reg_grid_search(X, y):
    # find optimal regularization strength
    # TODO: refactor for SGD
    cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'C': cs}]
    clf = LogisticRegression(solver='lbfgs',
                             multi_class='multinomial',
                             random_state=0,
                             verbose=1)
    gs = GridSearchCV(clf, param_grid,
                      scoring='accuracy',
                      cv=5,
                      verbose=1)
    gs = gs.fit(X, y)
    return gs.best_params_

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

def main(argv):
    if len(argv) == 2:
        mode = argv[0][2:]
        ngram_order = int(argv[1])
    else:
        if '--bow' in argv:
            ngram_order = 1
            mode = 'bow'
        elif '--feats' in argv:
            ngram_order = 1
            mode = 'feats'
        else:
            ngram_order = int(argv[0])
    print "ngram_order:", ngram_order
    print "features:", mode, '\n'
    if mode == 'bow':
        train_test_bow(ngram_order)
    elif mode == 'feats':
        train_test_feats()

if __name__ == '__main__':
    main(sys.argv[1:])

