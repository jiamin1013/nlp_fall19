#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separatiing business logic from other stuff.  (9/19/2019)

import logging
import math
import re
import sys

from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Set, Tuple, Union

#myimport
import numpy as np
import random

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.

Zerogram = Tuple[()]
Unigram = Tuple[str]
Bigram = Tuple[str, str]
Trigram = Tuple[str, str, str]
Ngram = Union[Zerogram, Unigram, Bigram, Trigram]
Vector = List[float]


BOS = "BOS"  # special word type for context at Beginning Of Sequence
EOS = "EOS"  # special word type for observed token at End Of Sequence
OOV = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL = "OOL"  # special word type for all Out-Of-Lexicon words
OOV_THRESHOLD = (
    3
)  # minimum number of occurrence for a word to be considered in-vocabulary


def get_tokens(file: Path):
    """Iterate over the tokens, saving a few layers of nesting."""
    with open(file) as corpus:
        for line in corpus:
            for z in line.split():
                yield z
    yield EOS  # Every file implicitly ends with EOS.


class LanguageModel:
    def __init__(self):
        self.tokens: Counter[Ngram] = Counter()  # the c(...) function.
        self.vocab: Optional[Set[str]] = None
        self.progress = 0  # To print progress.

    @classmethod
    def make(cls, smoother: str, lexicon: Path) -> "LanguageModel":
        """Factory pattern: Build the language model you need."""
        r = re.compile("^(.*?)-?([0-9.]*)$")
        m = r.match(smoother)

        lambda_: Optional[float]  # Type annotation only.

        if m is None or not m.lastindex:
            raise ValueError(f"Smoother regular expression failed for {smoother}")
        else:
            smoother_name = m.group(1).lower()
            if m.lastindex >= 2 and len(m.group(2)):
                lambda_arg = m.group(2)
                lambda_ = float(lambda_arg)
            else:
                lambda_ = None

        if lambda_ is None and smoother_name.find("add") != -1:
            raise ValueError(
                f"You must include a non-negative lambda value in smoother name {smoother}"
            )

        if smoother_name == "uniform":
            return UniformLanguageModel()
        elif smoother_name == "add":
            assert lambda_ is not None
            return AddLambdaLanguageModel(lambda_)
        elif smoother_name == "backoff_add":
            assert lambda_ is not None
            return BackoffAddLambdaLanguageModel(lambda_)
        elif smoother_name == "loglinear":
            assert lambda_ is not None
            return LogLinearLanguageModel(lambda_, lexicon)
        else:
            raise ValueError(f"Don't recognize smoother name {smoother_name}")

    def file_log_prob(self, corpus: Path) -> float:
        """Compute the log probability of the sequence of tokens in file.
        NOTE: we use natural log for our internal computation.  You will want to
        divide this number by log(2) when reporting log probabilities.
        """
        log_prob = 0.0
        x, y = BOS, BOS
        for z in get_tokens(corpus):
            prob = self.prob(x, y, z)
            log_prob += math.log(prob)
            x, y = y, z  # Shift over by one position.
        return log_prob

    def set_vocab_size(self, *files: Path) -> None:
        if self.vocab is not None:
            log.warning("Warning: vocabulary already set!")

        word_counts: Counter[str] = Counter()  # count of each word

        for file in files:
            for token in get_tokens(file):
                word_counts[token] += 1
                self.show_progress()
        sys.stderr.write("\n")  # done printing progress dots "...."

        vocab: Set[str] = set(w for w in word_counts if word_counts[w] >= OOV_THRESHOLD)
        vocab |= {  # Union equals
            OOV,
            EOS,
        }  # add EOS to vocab (but not BOS, which is never a possible outcome but only a context)

        self.vocab = vocab
        log.info(f"Vocabulary size is {self.vocab_size} types including OOV and EOS")

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    def count(self, x: str, y: str, z: str) -> None:
        """Count the n-grams.  In the perl version, this was an inner function.
        For now, I am just using a data member to store the found tri-
        and bigrams.
        """
        self._count_ngram((x, y, z))
        self._count_ngram((y, z))
        self._count_ngram((z,))
        self._count_ngram(())

    def _count_ngram(self, ngram: Ngram) -> None:
        """Count the n-gram; that is, increment its count in the model."""
        self.tokens[ngram] += 1

    def num_tokens(self, corpus: Path) -> int:
        """Give the number of tokens in the corpus, including EOS."""
        return sum(1 for token in get_tokens(corpus))

    def prob(self, x: str, y: str, z: str) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("Reimplement this in subclasses!")
        raise NotImplementedError(
            f"{class_name} is not implemented yet. (That's your job!)"
        )

    @classmethod
    def load(cls, source: Path) -> "LanguageModel":
        import pickle

        log.info(f"Loading model from {source}")
        with open(source, mode="rb") as f:
            return pickle.load(f)
        log.info(f"Loaded model from {source}")

    def save(self, destination: Path) -> None:
        import pickle

        log.info(f"Saving model to {destination}")
        with open(destination, mode="wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved model to {destination}")

    def replace_missing(self, token: str) -> str:
        assert self.vocab is not None
        if token not in self.vocab:
            return OOV
        return token

    def train(self, corpus: Path) -> List[str]:
        """Read the training corpus and collect any information that will be needed
        by the prob function later on.  Tokens are whitespace-delimited.

        Note: In a real system, you wouldn't do this work every time you ran the
        testing program. You'd do it only once and save the trained model to disk
        in some format.
        """
        log.info(f"Training from corpus {corpus}")
        # The real work:
        # accumulate the type and token counts into the global hash tables.

        # If vocab size has not been set, build the vocabulary from training corpus
        if self.vocab is None:
            self.set_vocab_size(corpus)

        # Clear out any previous training.
        self.tokens = Counter()

        # We save the corpus in memory to a list tokens_list.  Notice that we
        # prepended two BOS at the front of the list and appended an EOS at the end.  You
        # will need to add more BOS tokens if you want to use a longer context than
        # trigram.
        x, y = BOS, BOS  # Previous two words.  Initialized as "beginning of sequence"
        # count the BOS context

        self.tokens[(x, y)] = 1
        self.tokens[(y,)] = 1  # The syntax for a 1-element tuple in Python

        tokens_list = [x, y]  # the corpus saved as a list
        for z in get_tokens(corpus):
            z = self.replace_missing(z)
            self.count(x, y, z)
            self.show_progress()
            x, y = y, z  # Shift over by 1 word.
            tokens_list.append(z)
        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.tokens[()]} tokens")
        return tokens_list

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")


class UniformLanguageModel(LanguageModel):
    def prob(self, x: str, y: str, z: str) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(LanguageModel):
    def __init__(self, lambda_: float) -> None:
        super().__init__()

        if lambda_ < 0:
            log.error(f"Lambda value was {lambda_}")
            raise ValueError(
                "You must include a non-negative lambda value in smoother name"
            )
        self.lambda_ = lambda_

    def prob(self, x: str, y: str, z: str) -> float:
        assert self.vocab is not None
        x = self.replace_missing(x)
        y = self.replace_missing(y)
        z = self.replace_missing(z)
        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.
        return (self.tokens[x, y, z] + self.lambda_) / (
            self.tokens[x, y] + self.lambda_ * self.vocab_size
        )


class BackoffAddLambdaLanguageModel(LanguageModel):
    def __init__(self, lambda_: float) -> None:
        super().__init__()

        if lambda_ < 0:
            log.error(f"Lambda value was {lambda_}")
            raise ValueError(
                "You must include a non-negative lambda value in smoother name"
            )
        self.lambda_ = lambda_

    def prob(self, x: str, y: str, z: str) -> float:
        # TODO: Reimplement me!
        #The trigram should backoff to bigram, bigram to unigram
        #unigram to uniform
        assert self.vocab is not None
        x = self.replace_missing(x)
        y = self.replace_missing(y)
        z = self.replace_missing(z)
        uni=(self.tokens[z,]+self.lambda_)/(self.tokens[()]+self.lambda_*self.vocab_size)
        bi=(self.tokens[y,z]+self.lambda_*self.vocab_size*uni)/(self.tokens[y,]+self.lambda_*self.vocab_size)
        tri=(self.tokens[x,y,z]+self.lambda_*self.vocab_size*bi)/(self.tokens[x,y]+self.lambda_*self.vocab_size)
        return tri


class LogLinearLanguageModel(LanguageModel):
    def __init__(self, c: float, lexicon: Path) -> None:
        super().__init__()
        if c < 0:
            log.error(f"C value was {c}")
            raise ValueError("You must include a non-negative c value in smoother name")
        self.c: float = c
        self.vectors: Dict[str, Vector]
        self.dim: int
        self.vectors, self.dim = self._read_vectors(lexicon)

        self.X: Any = None
        self.Y: Any = None

    def _read_vectors(self, lexicon: Path) -> Tuple[Dict[str, Vector], int]:
        """Read word vectors from an external file.  The vectors are saved as
        arrays in a dictionary self.vectors.
        """
        with open(lexicon) as f:
            header = f.readline()
            dim = int(header.split()[-1])
            vectors: Dict[str, Vector] = {}
            for line in f:
                word, *arr = line.split()
                vectors[word] = [float(x) for x in arr]

        return vectors, dim

    def replace_missing(self, token: str) -> str:
        # substitute out-of-lexicon words with OOL symbol 
        assert self.vocab is not None
        if token not in self.vocab:
            token = OOV
        #Original
        if token not in self.vectors:
            token = OOL
        #modified to treat OOV as OOL too
        if token == OOV:
            token = OOL
        return token

    def prob(self, x: str, y: str, z: str) -> float:
        # TODO: Reimplement me!
        assert self.vocab is not None
        x=self.replace_missing(x)
        y=self.replace_missing(y)
        z=self.replace_missing(z)

        assert self.vectors is not None
        x = np.asarray(self.vectors[x],dtype="float").reshape((-1,1))
        y = np.asarray(self.vectors[y],dtype="float").reshape((-1,1))
        z = np.asarray(self.vectors[z],dtype="float").reshape((-1,1))
        numer = self.calNumer(x,y,z)
        denom = self.calDenom(x,y)
        triprob=float(numer/np.sum(denom))
        return triprob

    def calNumer(self, x: np.ndarray,y: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = x.T if x.shape[1] != 1 else x
        y = y.T if y.shape[1] != 1 else y
        z = z.T if z.shape[1] != 1 else z
        assert self.X is not None
        assert self.Y is not None
        X = np.asarray(self.X,dtype="float")
        Y = np.asarray(self.Y,dtype="float")
        numer=np.exp(np.dot(np.dot(x.T,X),z)+np.dot(np.dot(y.T,Y),z))
        return numer

    def calDenom(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        #a handy function to calculate the normalization constant
        #for the trigram conditional probability
        x = x.T if x.shape[1] != 1 else x
        y = y.T if y.shape[1] != 1 else y
        evoc = np.asarray([self.vectors[self.replace_missing(i)] for i in self.vocab],dtype="float")
        assert self.X is not None
        assert self.Y is not None
        X = np.asarray(self.X,dtype="float")
        Y = np.asarray(self.Y,dtype="float")
        denom=np.exp(np.dot(np.dot(x.T,X),evoc.T)+np.dot(np.dot(y.T,Y),evoc.T)) #shape is 1-by-V
        return denom

    def train(self, corpus: Path) -> List[str]:
        """Read the training corpus and collect any information that will be needed
        by the prob function later on.  Tokens are whitespace-delimited.

        Note: In a real system, you wouldn't do this work every time you ran the
        testing program. You'd do it only once and save the trained model to disk
        in some format.
        """
        tokens_list = super().train(corpus)
        # Train the log-linear model using SGD.

        # Initialize parameters
        self.X = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]
        self.Y = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]
        # Optimization hyperparameters
        gamma0 = 0.01  # initial learning rate, used to compute actual learning rate
        epochs = 10  # number of passes
        t = 0 # number of updates in total, my
        self.N = len(tokens_list) - 2  # number of training instances
        # ******** COMMENT *********
        # In log-linear model, you will have to do some additional computation at
        # this point.  You can enumerate over all training trigrams as following.
        #
        # for i in range(2, len(tokens_list)):
        #   x, y, z = tokens_list[i - 2], tokens_list[i - 1], tokens_list[i]
        #
        # Note1: self.c is the regularizer constant C
        # Note2: You can use self.show_progress() to log progress.
        #
        # **************************
        print("Training from corpus",corpus.name)
        print("Vocabulary size is {0} types including OOV and EOS".format(self.vocab_size))
        for ep in range(0,epochs):
            objective = 0.0
            for i in range(2,len(tokens_list)):
                print("i is {0}".format(i))
                xtk,ytk,ztk = tokens_list[i-2],tokens_list[i-1],tokens_list[i]
                x = np.asarray(self.vectors[xtk],dtype="float").reshape((-1,1))
                y = np.asarray(self.vectors[ytk],dtype="float").reshape((-1,1))
                z = np.asarray(self.vectors[ztk],dtype="float").reshape((-1,1))

                #log.info("Start optimizing.")
                gamma = gamma0/(1+gamma0*2*self.c/self.N*t)
                constant = np.sum(self.calDenom(x,y))
                #for X
                s1 = np.multiply(x,z.T) #broadcast
                s1 = np.dot(x,z.T)
                sum_zi = np.zeros(z.T.shape) #1-by-m
                for zi in self.vocab:
                    zi = self.vectors[self.replace_missing(zi)]
                    zi = np.asarray(zi,dtype="float").reshape((1,-1))
                    p = float(self.calNumer(x,y,zi)/constant)
                    sum_zi = np.add(sum_zi,p*zi)

#                foo = np.asarray([(float(self.calNumer(x,y,np.asarray(self.vectors[self.replace_missing(zi)],dtype="float").reshape((-1,1)))/constant)*z.T).squeeze().tolist() for zi in self.vocab]) #foo is p*zi, which is V-by-m
#                foo = np.sum(foo,axis=0,keepdims=True) # 1-by-m
#                print(foo)
#                input()
                s2 = np.multiply(x,sum_zi) #broadcast, to get j-by-m
                s3 = 2*self.c/self.N*np.asarray(self.X)
                dX = s1-s2-s3 #j-by-m
                #for Y
                s1 = np.multiply(y,z.T) #broadcast
                s2 = np.multiply(y,sum_zi) #broadcast, to get j-by-m
                s3 = 2*self.c/self.N*np.asarray(self.Y)
                dY = s1-s2-s3 #j-by-m
                #finite-difference check
#                a,b = np.random.randint(0,self.dim),np.random.randint(0,self.dim)
#                wsum = np.sum(np.asarray(self.X)**2+np.asarray(self.Y)**2)
#                obj = np.log(self.prob(xtk,ytk,ztk)-self.c/self.N*wsum)
#                X_nw = np.asarray(self.X)
#                X_nw[a][b] = X_nw[a][b]+1e-6
#                Y_nw = np.asarray(self.Y)
#                self.X = X_nw.tolist()
#                wsum = np.sum(X_nw**2+Y_nw**2)
#                lhs = np.log(self.prob(xtk,ytk,ztk)-self.c/self.N*wsum)
#                rhs = obj+1e-6*dX[a][b]
#                print(abs(lhs-rhs))
#                input()
                X_nw = np.asarray(self.X)+gamma*dX
                Y_nw = np.asarray(self.Y)+gamma*dY
                self.X = X_nw.tolist()
                self.Y = Y_nw.tolist()
#                wsum = np.sum(X_nw**2+Y_nw**2)
#                objective+=np.log(self.prob(xtk,ytk,ztk))-self.c/self.N*wsum
                t+=1
                self.show_progress()
#               wsum = np.sum(self.x**2+self.y**2)
#               objective+=np.log(self.prob(xtk,ytk,ztk))-self.c/self.n*wsum 

            wsum = np.sum(np.asarray(self.X)**2+np.asarray(self.Y)**2)
            for i in range(2,len(tokens_list)):
                xtk,ytk,ztk = tokens_list[i-2],tokens_list[i-1],tokens_list[i]
                objective+=np.log(self.prob(xtk,ytk,ztk))
            objective = 1/self.N*objective-self.c/self.N*wsum
            print("epoch {0}: F={1:.5f}".format(ep,objective))
        sys.stderr.write("\n")  # done printing progress dots "...."

        #####################
        # TODO: Implement your SGD here
        #####################

        log.info(f"Finished training on {self.tokens[()]} tokens")
        return tokens_list  # Not really needed, except to obey typing.

#ck = 0.0
#x = next(iter(self.vocab))
#y = next(iter(self.vocab))
#for z in self.vocab:
#    x = self.replace_missing(x)
#    y = self.replace_missing(y)
#    z = self.replace_missing(z)
#    ck += self.prob(x,y,z)
#print(ck)
#input()

