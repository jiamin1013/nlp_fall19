#! /usr/bin/env python3
import math
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("lexicon",type=str)
    parser.add_argument("anchor_word",type=str)
    args = parser.parse_args()
    return args

def readLexi(lexi):
    lexicon={}
    with open(lexi,'r') as infile:
        infile.readline() #burst first line
        for line in infile:
            foo = line.strip().split()
            lexicon[foo[0]] = [float(i) for i in foo[1:]]
    return lexicon

def cos_sim(a,b):
    a,b=np.array(a).reshape((1,-1)),np.array(b).reshape((1,-1))
    a = a/np.linalg.norm(a,ord=2)
    b = b/np.linalg.norm(b,ord=2)
    score = np.dot(a,b.T).tolist()
    return score[0][0]

def main():
    args = get_args()
    lexi,aword = args.lexicon,args.anchor_word
    lexicon = readLexi(lexi)
    avec = lexicon[aword]
    #rather inefficient implementation
    scores,words = [],[]
    for word in lexicon.keys():
        if word != aword:
            words.append(word)
            scores.append(cos_sim(avec,lexicon[word]))
    scores = np.array(scores)
    topten = scores.argsort()[-10:][::-1]
    print([words[int(ind)] for ind in topten])

if __name__=="__main__":
    main()

