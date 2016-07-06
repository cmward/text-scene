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


with open('datafiles/ff_log.txt', 'wa') as log:
    for layer_size in layer_sizes:
        for pool_mode in pool_modes:
            for label_set in label_sets:
                cv_scores = train(label_set=label_set,
                                  pool_mode=pool_mode,
                                  layer_sizes=layer_size,
                                  drop_unk=True,
                                  word_vecs='../../GoogleNews-vectors-negative300.bin')
                log.write(('layers: %s\t\tpool mode: %s\t\tlabel set: %s\t\t
                           acc: %.4f') % (layer_size,
                                          pool_mode,
                                          label_set,
                                          np.mean(cv_scores))
