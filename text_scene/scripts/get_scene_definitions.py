import sys
import glob
from nltk.corpus import wordnet as wn

"""
Usage:
    python get_scene_definitions.py <category file> <output file>
"""

nodef = []

def definition(string):
    try:
        return wn.synsets(string)[0].definition()
    except IndexError:  # word not in wordnet
        nodef.append(string)
        return 'CATEGORY NOT FOUND IN WORDNET'

def categories(category_file):
    with open(category_file) as reader:
        for line in reader:
            yield line.split()[0].split('/')[2]

def main(argv):
    with open(argv[1], 'w+') as writer:
        for category in categories(argv[0]):
            writer.write(category + '\t' + definition(category) + '\n')
        writer.write('\nNO DEFINITIONS:\n')
        for category in nodef:
            writer.write(category + '\n')

if __name__ == '__main__':
    main(sys.argv[1:])
