import sys
import os
import glob

"""
Usage:
    python get_aws_image_urls.py <upload_batches_dir> <output_file>
"""

root_url = 'http://s3.amazonaws.com/caption-scene/upload_batches'

def make_url(filename, root_url):
    return '{}/{}'.format(root_url, filename)

def main(argv):
    batches_dir = argv[0]
    output_file = argv[1]
    image_files = glob.glob(batches_dir + '/*/*.jpg')
    image_files = ['/'.join(f.split('/')[-2:]) for f in image_files]
    image_urls = [make_url(fname, root_url) for fname in image_files]
    with open(output_file, 'w+') as outf:
        for image_url in image_urls:
            outf.write(image_url + '\n')

if __name__ == '__main__':
    main(sys.argv[1:])
