import gc
import os
import sys
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from classification.train_feedforward import train

layer_sizes = [[128, 128],
               [256, 256],
               [512, 512],
               [1024, 1024],
               [256, 128]]
pool_modes = ['sum', 'max', 'mean']
label_sets = ['full']#, 'function', '3way']
activations = ['tanh', 'relu', 'leakyrelu', 'prelu', 'elu']
word_vecs = sys.argv[1]


with open('datafiles/experiments/ff_log.txt', 'wa') as log:
    log.write(time.ctime() + '\n')
    for layer_size in layer_sizes:
            for pool_mode in pool_modes:
                for activation in activations:
                    for label_set in label_sets:
                        drop_unk = False if label_set == '3way' else True
                        cv_scores = train(label_set=label_set,
                                          pool_mode=pool_mode,
                                          layer_sizes=layer_size,
                                          activation=activation,
                                          drop_unk=drop_unk,
                                          word_vecs=word_vecs,
                                          val_split=0.)
                        log.write(('%s\t%s\t%s\t%s\t%.4f\n') %
                            (str(layer_size),
                             pool_mode,
                             activation,
                             label_set,
                             np.mean(cv_scores)))
                        gc.collect()
