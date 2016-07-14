import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import StratifiedKFold
from preprocessing.data_utils import sentences_df, load_dataset
from paths import SENTENCES_CSV

def train_test_bow(ngram_order, batch_size=128, n_epoch=3):
    label_sets = ['full', 'function', '3way', 'in_out', 'man_nat']
    for label_set in label_sets:
        # need to drop unk for full/function
        if label_set in ['full', 'function']:
            df = sentences_df(labels=label_set, drop_unk=True)
        else:
            df = sentences_df(SENTENCES_CSV, labels=label_set, drop_unk=False)
        X, y, word2idx, l_enc = load_dataset(df, ngram_order=ngram_order)
        print "X shape: %s" % (X.shape,)
        print "y shape: %s" % (y.shape,)
        skf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=0)
        scores = []
        for (train, test) in skf:
            clf = None
            clf = SGDClassifier(loss='log',
                                alpha=0.001,
                                l1_ratio=0,
                                random_state=0)
            for epoch in range(n_epoch):
                X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
                n_batches = X_train.shape[0] // batch_size
                for minibatch_idx in range(n_batches):
                    clf.partial_fit(
                        X_train[minibatch_idx * batch_size : (minibatch_idx+1) * batch_size],
                        y_train[minibatch_idx * batch_size : (minibatch_idx+1) * batch_size],
                        classes=np.unique(y))
                print "Epoch: %d/%d Train acc: %.4f" \
                    % (epoch+1, n_epoch, clf.score(X_train, y_train))
            fold_score = clf.score(X_test, y_test)
            print "Fold acc: %.4f" % fold_score
            scores.append(fold_score)
        print '%s label mean cv accuracy: %.4f\n' % (label_set, np.mean(scores))

def train_test_feats():
    pass

def train_and_test_maxent(ngram_order=1, feats='bow'):
    print "ngram_order:", ngram_order
    print "features:", feats, '\n'
    if feats == 'bow':
        train_test_bow(ngram_order)
    elif feats == 'feats':
        train_test_feats()
