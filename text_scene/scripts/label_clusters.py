import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from nltk.corpus import wordnet as wn
from collections import defaultdict
from get_scene_definitions import categories

category_file = '../../data/places2_categories.txt'
labels = categories(category_file)
in_wn = set((l for l in labels if wn.synsets(l, wn.NOUN)))
label2idx = {w:i for i,w in enumerate(in_wn)}
idx2word = {i:w for w,i in label2idx.items()}

arena_syn = wn.synset('stadium.n.01')
yard_syn = wn.synset('yard.n.02')
fixed = {'arena': arena_syn,
         'yard': yard_syn}

def wn_sim_matrix(label2idx):
    sim = np.zeros((len(label2idx), len(label2idx)))
    for w1, i1 in label2idx.items():
        if w1 in fixed:
            syn1 = fixed[w1]
        else:
            syn1 = wn.synsets(w1, wn.NOUN)[0]
        for w2, i2 in label2idx.items():
            if w2 in fixed:
                syn2 = fixed[w2]
            else:
                syn2 = wn.synsets(w2, wn.NOUN)[0]
            sim[i1,i2] = syn1.lch_similarity(syn2)
    return sim

def standardize(sim):
    sc = StandardScaler()
    sim = sc.fit_transform(sim)
    return sim

def km_cluster(sim, n_clusters):
    km = KMeans(n_clusters=n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    sim_km = km.fit_predict(sim)
    return sim_km

def h_cluster(sim):
    row_clusters = linkage(pdist(sim, metric='euclidean'),
                           method='complete')
    return row_clusters

def dendrogram(row_clusters):
    labels = [w for w in label2idx.keys()]
    row_dendr = dendrogram(row_clusters,
                           labels=labels)
    plt.show()

def make_cluster_dict(sim_km):
    clusters = defaultdict(list)
    for i, cn in enumerate(sim_km):
        clusters[cn].append(idx2word[i])
    return clusters

def elbow(sim):
    distortions = []
    for i in range(1, 101):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(sim)
        distortions.append(km.inertia_)
    plt.plot(range(1, 101), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()
