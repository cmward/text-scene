import os
import sys
import time
import numpy as np
import pandas as pd
from collections import Counter
from keras.utils.np_utils import to_categorical, probas_to_classes
from keras.utils.layer_utils import print_summary
from sklearn.cross_validation import StratifiedKFold, train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.data_utils import (
    load_bin_vec,
    sentences_df,
    load_dataset,
    add_unknown_words,
    create_unk_labeled_instances
)
from corpus_stats.frequency import print_label_frequencies
from feedforward import FeedforwardNN, train_and_test_model
from paths import SENTENCES_CSV


# hyperparameters
emb_dim = 300
batch_size = 64
nb_epoch = 12
lr = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08

def train(label_set='full', pool_mode='max', layer_sizes=[256, 256],
          drop_unk=False, word_vecs=None, return_net=False, cv=5,
          label_unk=False):
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

    if label_unk:
        unk_df = pd.read_csv(SENTENCES_CSV)
        unk_df = unk_df[(unk_df.q3 == 'other_unclear')| (unk_df.q4 == 'other_unclear')]
        df2 = create_unk_labeled_instances(unk_df)
        df3 = sentences_df(label_unk=df2)
        X_unk, y_unk, _, _ = load_dataset(df3, pad=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y_orig, test_size=0.2, random_state=0)

        print "Training and testing without labeled unknown on %i samples" % len(X_train)
        nn1 = FeedforwardNN(vocab_size,
                            nb_labels,
                            emb_dim,
                            maxlen,
                            layer_sizes,
                            embedding_weights,
                            pool_mode=pool_mode)

        _, acc = train_and_test_model(nn1, X_train, y_train, X_test, y_test,
                                       batch_size, nb_epoch,
                                       lr, beta_1, beta_2, epsilon)

        print "Acc: %.4f" % acc

        print X_train.shape, X_unk.shape
        X_train = np.vstack((X_train, X_unk))
        y_train = np.vstack((y_train, to_categorical(y_unk)))
        print "Training and testing with labeled unknown on %i samples" % len(X_train)
        nn2 = FeedforwardNN(vocab_size,
                            nb_labels,
                            emb_dim,
                            maxlen,
                            layer_sizes,
                            embedding_weights,
                            pool_mode=pool_mode)

        _, acc = train_and_test_model(nn2, X_train, y_train, X_test, y_test,
                                       batch_size, nb_epoch,
                                       lr, beta_1, beta_2, epsilon)

        print "Acc: %.4f" % acc

    elif cv:
        skf = StratifiedKFold(y_orig, n_folds=cv, shuffle=True, random_state=0)
        cv_scores = []
        for i, (train, test) in enumerate(skf):
            start_time = time.time()
            nn = None
            nn = FeedforwardNN(vocab_size,
                               nb_labels,
                               emb_dim,
                               maxlen,
                               layer_sizes,
                               embedding_weights,
                               pool_mode=pool_mode)
            if i == 0:
                print_summary(nn.model.layers)
            nn, acc = train_and_test_model(nn, X[train], y[train], X[test], y[test],
                                           batch_size, nb_epoch,
                                           lr, beta_1, beta_2, epsilon)
            if return_net:
                d = {'X': X,
                     'y': y,
                     'word2idx': word2idx,
                     'l_enc': l_enc,
                     'y_binary': y_binary,
                     'labels': labels,
                     'nb_labels': nb_labels,
                     'maxlen': maxlen,
                     'emb_dim': emb_dim,
                     'vocab_size': vocab_size,
                     'embedding_weights': embedding_weights}
                return d, nn
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
            print "\nfold %i/5 - time: %.2f s - acc: %.4f on %i samples" % \
                (i+1, train_time, acc, len(test))
        print "Avg cv accuracy: %.4f" % np.mean(cv_scores)
