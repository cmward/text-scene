import sys
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import cross_val_score
from data_utils import load_data, load_dataset
from paths import SENTENCES_CSV

if len(sys.argv) == 2:
    ngram_order = int(sys.argv[1])
else:
    ngram_order = 1
print "ngram order:", ngram_order

label_sets = ['full', 'function', 'in_out', 'man_nat']
for label_set in label_sets:
    df = load_data(SENTENCES_CSV, labels=label_set)
    X, y, word2idx, l_enc = load_dataset(df, ngram_order=ngram_order)
    clf = BernoulliNB()
    scores = cross_val_score(clf, X, y, cv=5, verbose=1)
    print '%s label mean cv accuracy: %.2f' % (label_set, np.mean(scores))
