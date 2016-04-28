import sys
from mturk import mturk_hits
from classification.data_utils import make_datadict, write_sentence_csv
from paths import CAPTIONS_FILE, SENTENCES_CSV

def main(argv):
    if argv[1] == '--mturk':
        mturk_hits.main(argv[1:])
    elif argv[1] == '--sentcsv':
        datadict = make_datadict(argv[2])
        write_sentence_csv(datadict, CAPTIONS_FILE, SENTENCES_CSV)
    elif argv[1] == '--cnn':
        from classification import cnn
        model_type = argv[2]
        label_set = argv[3]
        word_vecs = argv[4]
        cnn.main(model_type=model_type,
                 label_set=label_set,
                 word_vecs=word_vecs)

if __name__ == '__main__':
    main(sys.argv)
