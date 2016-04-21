import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import cross_val_score
from load_data import load_data, load_dataset
from paths import SENTENCES_CSV

df = load_data(SENTENCES_CSV)
X, y, word2id, l_enc = load_dataset(df, ngram_order=1)
clf = BernoulliNB()
scores = cross_val_score(clf, X, y, cv=10)
print 'Full label mean cv accuracy:', np.mean(scores)

df = load_data(SENTENCES_CSV, labels='function')
X, y, word2id, l_enc = load_dataset(df, ngram_order=1)
clf = BernoulliNB()
scores = cross_val_score(clf, X, y, cv=10)
print 'function/nat label mean cv accuracy:', np.mean(scores)

df = load_data(SENTENCES_CSV, labels='in_out')
X, y, word2id, l_enc = load_dataset(df, ngram_order=1)
clf = BernoulliNB()
scores = cross_val_score(clf, X, y, cv=10)
print 'in/out label mean cv accuracy:', np.mean(scores)

df = load_data(SENTENCES_CSV, labels='man_nat')
X, y, word2id, l_enc = load_dataset(df, ngram_order=1)
clf = BernoulliNB()
scores = cross_val_score(clf, X, y, cv=10)
print 'man/nat label mean cv accuracy:', np.mean(scores)
