import os
from os.path import join as pjoin

ROOT = os.path.abspath('.')  # text_scene directory
CAPTIONS_FILE = os.path.abspath(
    '../../data/Flickr30k/flickr30k/results_20130124.token')
DATAFILES = pjoin(ROOT, 'datafiles')
SENTENCES_CSV = pjoin(DATAFILES, 'sentences.csv')
ANNOTATED_IMGS_FILE = pjoin(DATAFILES, 'annotated_images.txt')
REJECTED_IMGS_FILE = pjoin(DATAFILES, 'rejected_images.txt')
MTURK_RESULTS_CSV = pjoin(DATAFILES, 'mturk_results.csv')
GOLD_MTURK_RESULTS_CSV = pjoin(DATAFILES, 'gold_mturk_results.csv')
MAJORITY_MTURK_RESULTS_CSV = pjoin(DATAFILES, 'majority_mturk_results.csv')
COMBINED_MTURK_RESULTS_CSV = pjoin(DATAFILES, 'combined_mturk_results.csv')
IMG_URLS = pjoin(DATAFILES, 'image_urls.csv')
KEY_FILE = os.path.abspath('../../data/rootkey.csv')
REDO_IMGS_FILE = pjoin(DATAFILES, 'redo_images.txt')
