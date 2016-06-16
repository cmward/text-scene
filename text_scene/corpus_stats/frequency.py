import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def word_freq_by_labels(df):
    """
    Given a dataframe of labeled sentences, return
    a dictionary of word frequencies given a label.

    Parameters
    ----------
    df: dataframe containing sentences and labels

    Returns
    -------
    freqdict: dictionary mapping each label to a
        dictionary mapping words to frequencies.
    """
    freqdict = defaultdict(Counter)
    for _, row in df.iterrows():
        label = row['label']
        tokens = row['sentence'].split()
        freqdict[label].update(tokens)
    for label, count_dict in freqdict.items():
        total = sum(count_dict.values())
        freqs = Counter({k: v / float(total) for k,v in count_dict.items()})
        freqdict[label] = freqs
    return freqdict

def label_frequencies(data):
    """
    Given a dataset, return a dictionary mapping labels to label frequencies.

    `data` is either a single dataframe or a tuple (y, l_enc) where y is a
    numpy array of labels and l_enc is a LabelEncoder.
    """
    if isinstance(data, pd.DataFrame):
        counts = defaultdict(float)
        for i, row in data.iterrows():
            counts[row['label']] += 1
    elif isinstance(data, tuple):
        y = data[0]
        l_enc = data[1]
        bincounts = np.bincount(y)
        counts = {l_enc.inverse_transform(i): float(count)
                 for i, count in enumerate(bincounts)}
    total = sum(counts.values())
    freqs = {k: (v/total) for k,v in counts.items()}
    sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    return sorted_freqs

def print_label_frequencies(data):
    freqs = label_frequencies(data)
    for (label, freq) in freqs:
        print "%s: %.2f" % (label, freq)
