from math import ceil
import csv

image_urls_file = '../../data/image_urls.txt'

def image_urls(urls_file):
    with open(urls_file) as f:
        for line in f:
            yield line.strip()

def make_data_files(urls, batch_size=None):
    urls = list(image_urls(image_urls_file))
    n_urls = len(urls)
    if batch_size is None:
        batch_size = n_urls
    n_batches = ceil(float(n_urls) / batch_size)
    batches = [urls[i * batch_size: (i+1) * batch_size]
               for i in range(int(n_batches))]
    for i,batch in enumerate(batches):
        filename = 'scene_rec_{}.csv'.format(i+1)
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(['image_url'])
            for url in batch:
                writer.writerow([url])

if __name__ == '__main__':
    make_data_files(image_urls_file)
