import sys
import argparse
from preprocessing.data_utils import (
    make_datadict,
    write_sentence_csv,
    majority_vote_dict,
    write_majority_vote_csv,
    write_no_majority_list,
    combine_csvs,
    fleiss_kappa,
    unique_workers
)
from paths import (
    CAPTIONS_FILE,
    SENTENCES_CSV,
    COMBINED_MTURK_RESULTS_CSV,
    GOLD_MTURK_RESULTS_CSV,
    MTURK_RESULTS_CSV,
    MAJORITY_MTURK_RESULTS_CSV,
    REDO_IMGS_FILE
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mturk', type=str, required=False)
    parser.add_argument('--sentcsv', '-s', type=str, required=False)
    parser.add_argument('--classify', '-c', type=str, required=False)
    parser.add_argument('--approve', '-a', action='store_true', required=False)
    parser.add_argument('--fleiss', type=str, required=False)
    parser.add_argument('--workers', type=str, required=False)
    parser.add_argument('--outfile', '-o', type=str, required=False)
    parser.add_argument('--logfile', '-l', type=str, required=False)
    parser.add_argument('--nimages', '-n', type=int, required=False)
    parser.add_argument('--redolog', type=str, required=False)
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--labelset', type=str, required=False)
    parser.add_argument('--dropunk', action='store_true', required=False)
    parser.add_argument('--wordvecs', type=str, required=False)
    parser.add_argument('--poolmode', type=str, required=False)
    parser.add_argument('--layersizes', type=int, nargs='+', required=False)
    parser.add_argument('--ngram', type=int, required=False)
    parser.add_argument('--feats', '-f', type=str, required=False)
    args = parser.parse_args()

    if args.mturk == 'hits':
        from mturk import mturk_hits
        mturk_hits.main(make_hits=True, log_file=args.logfile,
                        n_images=args.nimages)
    if args.mturk == 'redo':
        from mturk import mturk_hits
        mturk_hits.main(redo_hits=True, redo_log=args.redolog, log_file=args.logfile)
    elif args.mturk == 'approve':
        from mturk import mturk_hits
        mturk_hits.main(approve=True, outfile=args.outfile,
                        log_file=args.logfile)
    elif args.mturk == 'fleiss':
        print "Fleiss kappa:", fleiss_kappa(args.outfile, args.labelset)
    elif args.mturk == 'workers':
        print unique_workers(args.outfile)


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

    elif args.classify == 'ff':
        from classification import train_feedforward
        label_set = args.labelset
        drop_unk = args.dropunk
        word_vecs = args.wordvecs
        pool_mode = args.poolmode
        layer_sizes = args.layersizes
        train_feedforward.train(label_set,
                                pool_mode=pool_mode,
                                layer_sizes=layer_sizes,
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
        if args.ngram:
            nb.train_and_test_nb(ngram_order=args.ngram-order)
        else:
            nb.train_and_test_nb()

    elif args.sentcsv == 'combined':
        # majority vote on combined for gold annotation of nonmajority
        combine_csvs(GOLD_MTURK_RESULTS_CSV, MTURK_RESULTS_CSV,
                     COMBINED_MTURK_RESULTS_CSV)
        write_majority_vote_csv(COMBINED_MTURK_RESULTS_CSV,
                                MAJORITY_MTURK_RESULTS_CSV)
        datadict = make_datadict(MAJORITY_MTURK_RESULTS_CSV)
        write_sentence_csv(datadict, CAPTIONS_FILE, SENTENCES_CSV)

    elif args.sentcsv == 'gold':
        datadict = make_datadict(GOLD_MTURK_RESULTS_CSV)
        write_sentence_csv(datadict, CAPTIONS_FILE, SENTENCES_CSV)

    elif args.sentcsv == 'mturk':
        datadict = make_datadict(MTURK_RESULTS_CSV)
        write_sentence_csv(datadict, CAPTIONS_FILE, SENTENCES_CSV)
