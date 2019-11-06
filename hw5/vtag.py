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

class HMM():
    def __init__(self):
        self.ptran = [] #ptran(0,j) will be initial probabilities
        self.pemis = {}
        self.tag_dict = {}
        self.tag2int = {}
        self.int2tag = {}
        self.num_states = None

    #update parameters from unsmoothed counts from train file 
    def count_estimate(self,train_file):
        transition,emission,states = {},{},{}
        ptag = None
        with open(train_file) as f:
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
                    if word not in self.tag_dict.keys():
                        self.tag_dict[word] = [tag]
                    elif tag not in self.tag_dict[word]:
                        self.tag_dict[word].append(tag)
                ptag = tag
        #assigning
        self.num_states = len(states.keys())
        #swap "###" and position 0, so that "###" will be mapped to 0
        new_keys = list(states.keys())
        foo,fooidx = new_keys[0],new_keys.index("###")
        new_keys[0],new_keys[fooidx] = "###",foo
        self.tag2int = dict(zip(new_keys,range(0,len(states.keys()))))
        self.int2tag = {v:k for k,v in self.tag2int.items()}
        self.ptran = np.zeros([self.num_states,self.num_states])
        for tran in transition.keys():
            i,j = self.tag2int[tran[0]],self.tag2int[tran[1]]
            denom = states[tran[0]] if i!=0 else states[tran[0]]-1
            self.ptran[i,j] = np.log(transition[tran])-np.log(denom)
        for emis in emission.keys():
            self.pemis[emis] = np.log(emission[emis])-np.log(states[emis[1]])
        assert(len(self.tag2int.keys())==self.num_states)

    #assumes input as one sentence, or a list of words
    #starting with ###, end before ###
    def viterbi_decode(self,sentence):
        #init
        bkptr = np.zeros([self.num_states,len(sentence)],dtype=np.int)
        bkptr[:,0] = 0 #starts with ###
        viter = np.full((self.num_states,len(sentence)),np.NINF)
        viter[:,0] = 0 #prob of generating ### is 1, in log domain is 0
        pretags = ["###"]
        all_states = list(self.tag2int.keys())
        all_states.remove('###')
        #algo
        for i in range(1,len(sentence)):
            wi,wI = sentence[i], sentence[i-1]
            #consider all states if wi is unknown
            curtags = self.tag_dict[wi] if wi in self.tag_dict.keys() else all_states
            for tagi in curtags:
                posi = self.tag2int[tagi]
                for tagI in pretags:
                    posI = self.tag2int[tagI]
                    pp = self.ptran[posI,posi]+self.pemis[(wi,tagi)]
                    vi = viter[posI,i-1]+pp
                    if vi > viter[posi,i]:
                        viter[posi,i] = vi
                        bkptr[posi,i] = posI
            pretags = curtags
        #handle last emission
        nT,best = 0,np.NINF
        for n in range(self.num_states):
                prob = viter[n,-1]+self.ptran[n,0]
                if prob > best:
                    nT,best = n,prob
        best_seq = [""]*(len(sentence)+1)
        best_seq[-1] = "###"
        for i in range(len(sentence)-1,-1,-1):
            tag = self.int2tag[nT]
            best_seq[i] = tag
            nT = bkptr[nT,i]
        return best_seq

    def eval_acc(self,seq_pred,seq_gold,sent):
        sent = sent[1:]
        seq_pred = np.asarray(seq_pred[1:-1],dtype=str)
        seq_gold = np.asarray(seq_gold[1:],dtype=str)
        #overall accuracy
        acc_all = np.sum(seq_pred==seq_gold)/len(seq_gold)*100
        #known accuracy
        mask = np.asarray([i in self.tag_dict.keys() for i in sent],dtype=bool)
        acc_know = np.sum(seq_pred[mask] == seq_gold[mask])/np.sum(mask)*100
        acc_unk = np.sum(seq_pred[~mask] == seq_gold[~mask])/np.sum(~mask)*100 if np.sum(~mask) !=0 else 0
        print("Tagging accuracy (Viterbi decoding): {0:2.2f}%\t(known: {1:2.2f}%  novel: {2:2.2f}%)".format(acc_all,acc_know,acc_unk))
        return

def readsents(test_file):
    #assume first line is "###"
    sent,sents = [], []
    seq,seqs = [], []
    flag = 1
    with open(test_file) as f:
        for line in f:
            word,tag = line.strip().split('/')
            if word == "###" and flag != 1:
                sents.append(sent)
                sent = [word]
                seqs.append(seq)
                seq = [tag]
            else:
                sent.append(word)
                seq.append(tag)
            flag = 0
    return sents,seqs

def main():
    args = get_args()
    filetrn,filetst = args.filetrn,args.filetst
    ichmm = HMM()
    ichmm.count_estimate(filetrn)
    sents,seqs = readsents(filetst)
    for i in range(len(sents)):
        sent,tag = sents[i],seqs[i]
        pred = ichmm.viterbi_decode(sent)
        ichmm.eval_acc(pred,tag,sent)
#    print(testsents)

if __name__ == "__main__":
    main()
