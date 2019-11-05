#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.special import logsumexp
from typing import NamedTuple
import random
import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filetrn",type=str,
                        help="Training file")
    parser.add_argument("filetst",type=str,
                        help="Testing file")
#    parser.add_argument("sent",type=str,
#                        help="sentence file")
    args = parser.parse_args()
    return args

def count_estimate(file):
    transition,emission,states = {},{},{}
    ptag = None
    with open(file) as f:
        for line in f:
            word,tag = line.strip().split('/')
            if tag not in states.keys():
                states[tag] = 1
            else:
                states[tag] += 1
            if (word,tag) not in emission.keys():
                emission[(word,tag)] = 1
            else:
                emission[(word,tag)] += 1
            if ptag is not None:
                if (ptag,tag) not in transition.keys():
                    transition[(ptag,tag)] = 1
                else:
                    transition[(ptag,tag)] += 1
            ptag = tag
    pinit,ptran,pemis = {},{},{}
    for tran in transition.keys():
        if tran[0] != "###":
            ptran[tran] = transition[tran]/states[tran[0]]
        else:
            if tran[1] not in pinit.keys():
                pinit[tran[1]] = 1
            else:
                pinit[tran[1]] += 1
    pinit = dict(zip(pinit.keys(), map(lambda x: x/sum(pinit.values()),pinit.values())))
    for emis in emission.keys():
        pemis[emis] = emission[emis]/states[emis[1]]
    return pinit,ptran,pemis
#TODO
def viterbi_decode():

def main():
    args = get_args()
    filetrn,filetst = args.filetrn,args.filetst
    pinit,ptran,pemis = count_estimate(filetrn)
    print(pinit)
    print(ptran)
    print(pemis)
    input()

if __name__ == "__main__":
    main()



#
#class wordtag(NamedTuple):
#    word: str
#    tag: str
#
#class ptagtag(NamedTuple):
#    word: str
#    tag: str
