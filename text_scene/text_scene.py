import sys
import argparse

def main(argv):
    # argparse:
    #   --classification/--mturk/--preprocessing
    if argv[1] == '--mturk':
        from mturk import mturk_hits
        mturk_hits.main(argv[1:])
    elif argv[1] == '--sentcsv':
        from mturk import mturk_hits
        from paths import CAPTIONS_FILE, SENTENCES_CSV
        from classification.data_utils import make_datadict, write_sentence_csv
        datadict = make_datadict(argv[2])
        write_sentence_csv(datadict, CAPTIONS_FILE, SENTENCES_CSV)
    elif argv[1] == '--cnn':
        from classification import train_cnn
        model_type = argv[2]
        label_set = argv[3]
        drop_unk = bool(argv[4])
        word_vecs = argv[5]
        train_cnn.train(model_type=model_type,
                        label_set=label_set,
                        drop_unk=drop_unk,
                        word_vecs=word_vecs)

if __name__ == '__main__':
    main(sys.argv)
