import sys
import time
import numpy as np
from train_feedforward import train

layer_sizes = [[128, 128],
               [256, 256],
               [512, 512],
               [1024, 1024],
               [256, 128],
               [512, 256],
               [512, 128],
               [1024, 512],
               [1024, 256],
               [1024, 128],
               [128, 128, 128],
               [256, 256, 256],
               [512, 512, 512],
               [1024, 1024, 1024],
               [1024, 512, 256],
               [512, 256, 128],
               [512, 512, 256]]
pool_modes = ['sum', 'max', 'mean', 'concat']
label_sets = ['full', 'function', '3way']
activations = ['tanh', 'relu', 'leakyrelu', 'prelu', 'elu']
word_vecs = sys.argv[1]


with open('datafiles/ff_log.txt', 'wa') as log:
    log.write(time.ctime() + '\n')
    for layer_size in layer_sizes:
            for pool_mode in pool_modes:
                for activation in activations:
                    for label_set in label_sets:
                        cv_scores = train(label_set=label_set,
                                          pool_mode=pool_mode,
                                          layer_sizes=layer_size,
                                          activation=activation,
                                          drop_unk=True,
                                          word_vecs=word_vecs)
                        log.write(('%s\t%s\t%s\t%s\t%.4f\n') %
                            (str(layer_size),
                             pool_mode,
                             activation,
                             label_set,
                             np.mean(cv_scores)))
