import numpy as np
import caffe

cafferoot = '/Users/chris/bin/caffe/'

mu = np.array([105.908874512, 114.063842773, 116.282836914])

net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(277,277), mean=mu,
                       raw_scale=255, channel_swap=(2,1,0))
