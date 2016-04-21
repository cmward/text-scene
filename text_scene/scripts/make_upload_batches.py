import glob
import os
from shutil import copyfile

image_dir = '../data/Flickr30k/flickr30k-images'
image_files = glob.glob(image_dir + '/*.jpg')

prev_i = 0
for i in xrange(len(image_files)):
    if i % 1000 == 0 and i != 0:
        directory = os.path.abspath(
            '../data/Flickr30k/upload_batches/{}_{}'.format(prev_i, i))
        if not os.path.exists(directory):
            os.makedirs(directory)
            print 'Copying {} - {}'.format(prev_i, i)
            for image in image_files[prev_i:i]:
                copyfile(image, os.path.join(directory, os.path.basename(image)))
        else:
            print '{} - {} already copied'.format(prev_i, i)
        prev_i = i
directory = os.path.abspath(
    '../data/Flickr30k/upload_batches/{}_{}'.format(prev_i, len(image_files)))
if not os.path.exists(directory):
    os.makedirs(directory)
for image in image_files[prev_i:]:
    copyfile(image, os.path.join(directory, os.path.basename(image)))
