import sys
from mturk import mturk_hits
from classification import load_data
from paths import CAPTIONS_FILE, SENTENCES_CSV

def main(argv):
    if argv[1] == '--mturk':
        mturk_hits.main(argv[1:])
    elif argv[1] == '--sentcsv':
        datadict = load_data.make_datadict(argv[2])
        load_data.write_sentence_csv(datadict, CAPTIONS_FILE, SENTENCES_CSV)

if __name__ == '__main__':
    main(sys.argv)
