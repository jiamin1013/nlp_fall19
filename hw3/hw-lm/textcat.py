#!/usr/bin/env python3
"""
Computes the log probability of the sequence of tokens in file,
according to a trigram model.  The training source is specified by
the currently open corpus, and the smoothing method used by
prob() is polymorphic.
"""
import argparse
import logging
from pathlib import Path

try:
    # Numpy is your friend. Not *using* it will make your program so slow.
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    #
    # We made this easier by including the environment file in this folder.
    # Install Miniconda, then create and activate the provided environment.
    import numpy as np
except ImportError:
    print("\nERROR! Try installing Miniconda and activating it.\n")
    raise


from Probs import LanguageModel

TRAIN = "TRAIN"
TEST = "TEST"

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def get_model_filename(smoother: str, lexicon: Path, train_file: Path) -> Path:
    return Path(f"{smoother}_{lexicon.name}_{train_file.name}.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices={TRAIN, TEST}, help="execution mode")
    parser.add_argument(
        "smoother",
        type=str,
        help="""Possible values: uniform, add1, backoff_add1, backoff_wb, loglinear1
  (the "1" in add1/backoff_add1 can be replaced with any real λ >= 0
   the "1" in loglinear1 can be replaced with any C >= 0 )
""",
    )
    parser.add_argument(
        "lexicon",
        type=Path,
        help="location of the word vector file; only used in the loglinear model",
    )
    parser.add_argument("train_file", type=Path, nargs=2, help="location of the training corpus")
    parser.add_argument("prior",type=float,nargs="?",help="prior probability of gen")
    parser.add_argument("test_files", type=Path, nargs="*")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()

    # Sanity-check the configuration.
    if args.mode == "TRAIN" and args.test_files:
        parser.error("Shouldn't see test files when training.")
    elif args.mode == "TEST" and not args.test_files:
        parser.error("No test files specified.")
    elif args.mode =="TEST" and not args.prior:
        parser.error("No prior specified.")
    elif args.mode == "TEST" and (args.prior < 0 or args.prior >1):
        parser.error("Prior has to be a probability.")

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    model_path = [get_model_filename(args.smoother, args.lexicon, tf) for tf in args.train_file]
    if args.mode == TRAIN:
        log.info("Training...")
        lm1 = LanguageModel.make(args.smoother, args.lexicon)
        lm2 = LanguageModel.make(args.smoother,args.lexicon)
        lm1.set_vocab_size(args.train_file[0],args.train_file[1])
        lm2.set_vocab_size(args.train_file[0],args.train_file[1])
        lm1.train(args.train_file[0])
        lm2.train(args.train_file[1])
        lm1.save(destination=model_path[0])
        lm2.save(destination=model_path[1])
    elif args.mode == TEST:
        log_prior_tar = np.log2(args.prior) #log domain for easy manipulation
        log_prior_ntar = np.log2(1-args.prior)
        log.info("Testing...")
        lm1 = LanguageModel.load(model_path[0]) #gen model
        lm2 = LanguageModel.load(model_path[1]) #spam model

        # We use natural log for our internal computations and that's
        # the kind of log-probability that fileLogProb returns.
        # But we'd like to print a value in bits: so we convert
        # log base e to log base 2 at print time, by dividing by log(2).

        log.info("Printing file log-likelihoods.")
        total_log_prob_tar,total_log_prob_ntar,tar_counts = 0.0,0
        tar_name,ntar_name = args.train_file[0].name,args.train_file[1].name
        for test_file in args.test_files:
            log_prob_tar = lm1.file_log_prob(test_file) / np.log(2) + log_prior_tar
            total_log_prob_tar += log_prob_tar
            log_prob_ntar = lm2.file_log_prob(test_file) / np.log(2) + log_prior_ntar
            total_log_prob_ntar += log_prob_ntar
            if log_prob_tar > log_prob_ntar:
                tar_counts+=1
                print("{0}\t{1}".format(tar_name,test_file))
            else:
                print("{0}\t{1}".format(ntar_name,test_file))
        ll = len(args.test_files)
        print("{0} files were probably {1} ({2:.2f}%)".format(tar_counts,tar_name,tar_counts/ll*100))
        print("{0} files were probably {1} ({2:.2f}%)".format(ll-tar_counts,ntar_name,(ll-tar_counts)/ll*100))

        token1=sum(lm1.num_token(args.test_files[0]))
        token2=sum(lm1.num_token(args.test_files[1]))
        print(f"Overall cross-entropy estimating from target model:\t{-total_log_prob / total_tokens:.5f}")


        #total_tokens = sum(lm.num_tokens(test_file) for test_file in args.test_files)
        #print(f"Overall cross-entropy:\t{-total_log_prob / total_tokens:.5f}")
    else:
        raise ValueError("Inappropriate mode of operation.")


if __name__ == "__main__":
    main()

