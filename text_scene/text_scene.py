import sys
import argparse
from preprocessing.data_utils import (
    make_datadict,
    write_sentence_csv,
    majority_vote_dict,
    write_majority_vote_csv,
    write_rejected_no_majority_list,
    combine_csvs
)
from mturk import mturk_hits
from paths import (
    CAPTIONS_FILE,
    SENTENCES_CSV,
    COMBINED_MTURK_RESULTS_CSV,
    GOLD_MTURK_RESULTS_CSV,
    MTURK_RESULTS_CSV,
    MAJORITY_MTURK_RESULTS_CSV
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mturk', type=str, required=False)
    parser.add_argument('--sentcsv', '-s', type=str, required=False)
    parser.add_argument('--classify', '-c', type=str, required=False)
    parser.add_argument('--approve', '-a', action='store_true', required=False)
    parser.add_argument('--outfile', '-o', type=str, required=False)
    parser.add_argument('--logfile', '-l', type=str, required=False)
    parser.add_argument('--nimages', '-n', type=int, required=False)
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--labelset', type=str, required=False)
    parser.add_argument('--dropunk', action='store_true', required=False)
    parser.add_argument('--wordvecs', type=str, required=False)
    parser.add_argument('--ngram', type=int, required=False)
    parser.add_argument('--feats', '-f', type=str, required=False)
    args = parser.parse_args()

    if args.mturk == 'hits':
        mturk_hits.main(make_hits=True, log_file=args.logfile,
                        n_images=args.nimages)
    elif args.mturk == 'approve':
        mturk_hits.main(approve=True, outfile=args.outfile,
                        log_file=args.logfile)

    elif args.classify == 'cnn':
        from classification import train_cnn
        model_type = args.model
        label_set = args.labelset
        drop_unk = args.dropunk
        word_vecs = args.wordvecs
        train_cnn.train(model_type=model_type,
                        label_set=label_set,
                        drop_unk=drop_unk,
                        word_vecs=word_vecs)

    elif args.classify == 'maxent':
        from classification import maxent
        if args.feats == 'bow':
            if args.ngram:
                maxent.train_and_test_maxent(ngram_order=args.ngram,
                                             feats=args.feats)
            else:
                maxent.train_and_test_maxent()
        else:
            train_and_test_maxent(feats=args.feats)

    elif args.classify == 'nb':
        from classification import nb
        if args.ngram_order:
            nb.train_and_test_nb(ngram_order=args.ngram-order)
        else:
            nb.train_and_test_nb()

    elif args.sentcsv == 'combined':
        write_majority_vote_csv(GOLD_MTURK_RESULTS_CSV,
                                           MAJORITY_MTURK_RESULTS_CSV)
        combine_csvs(GOLD_MTURK_RESULTS_CSV, MAJORITY_MTURK_RESULTS_CSV,
                     COMBINED_MTURK_RESULTS_CSV)
        datadict = make_datadict(COMBINED_MTURK_RESULTS_CSV)
        write_sentence_csv(datadict, CAPTIONS_FILE, SENTENCES_CSV)

    elif args.sentcsv == 'gold':
        datadict = make_datadict(GOLD_MTURK_RESULTS_CSV)
        write_sentence_csv(datadict, CAPTIONS_FILE, SENTENCES_CSV)

    elif args.sentcsv == 'mturk':
        datadict = make_datadict(MTURK_RESULTS_CSV)
        write_sentence_csv(datadict, CAPTIONS_FILE, SENTENCES_CSV)

