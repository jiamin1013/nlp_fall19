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
        self.states = {}
        self.num_states = None
        #smoothing
        self.lambda_emis = 0
        self.lambda_tran = 0

    def set_lambda(self,lambda_emis,lambda_tran=0):
        self.lambda_emis = lambda_emis
        self.lambda_tran = lambda_tran

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
        self.states = states
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
            #TODO smoothing?
            numer = transition[tran]
            self.ptran[i,j] = np.log(numer)-np.log(denom)
        #smoothing for emis?
        if self.lambda_emis != 0:
            self.tag_dict["OOV"] = list(self.tag2int.keys())[1:] #exclude ###
        for emis in emission.keys():
            numer = emission[emis]+self.lambda_emis
            denom = states[emis[1]]+self.lambda_emis*(len(self.tag_dict.keys())-1)
            self.pemis[emis] = np.log(numer) - np.log(denom)
        for state in states.keys():
            if state != "###" and self.lambda_emis != 0:
                numer = emission[emis]+self.lambda_emis
                denom = states[emis[1]]+self.lambda_emis*(len(self.tag_dict.keys())-1)
                self.pemis[("OOV",state)] = np.log(numer)-np.log(denom)
        assert(len(self.tag2int.keys())==self.num_states)

    #assumes input as one sentence, or a list of words
    #starting with ###, end before ###
    def viterbi_decode(self,sentence):
        #init
        bkptr = np.zeros([self.num_states,len(sentence)],dtype=np.int)
        bkptr[:,0] = 0 #starts with ###
        viter = np.full((self.num_states,len(sentence)),np.NINF)
        viter[:,0] = 0 #prob of generating ### is 1, in log domain is 0
        #all_states = list(self.tag2int.keys())
        #all_states.remove('###')
        pretags, wI = ["###"],"###"
        #algo
        for i in range(1,len(sentence)):
            #consider all states if wi is unknown
            wi = sentence[i] if sentence[i] in self.tag_dict.keys() else "OOV"
            curtags = self.tag_dict[wi]
            for tagi in curtags:
                posi = self.tag2int[tagi]
                for tagI in pretags:
                    posI = self.tag2int[tagI]
                    pp = self.ptran[posI,posi]+self.pemis[(wi,tagi)]
                    vi = viter[posI,i-1]+pp
                    if vi > viter[posi,i]:
                        viter[posi,i] = vi
                        bkptr[posi,i] = posI
            pretags,wI = curtags,wi
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

    def count_acc(self,seq_pred,seq_gold,sent):
        mask = np.asarray([i in self.tag_dict.keys() for i in sent],dtype=bool)
        know_correct = np.sum(seq_pred[mask]==seq_gold[mask])
        unk_correct = np.sum(seq_pred[~mask]==seq_gold[~mask])
        know_all = np.sum(mask)
        unk_all = np.sum(~mask)
#        acc_unk = np.sum(seq_pred[~mask] == seq_gold[~mask])/np.sum(~mask)*100 if np.sum(~mask) !=0 else 0
#        print("Tagging accuracy (Viterbi decoding): {0:2.2f}%\t(known: {1:2.2f}%  novel: {2:2.2f}%)".format(acc_all,acc_know,acc_unk))
        return know_correct,unk_correct,know_all,unk_all

    def eval_acc(self,seqs_pred,seqs_gold,sents):
        kcc,unkcc,kaa,unkaa = 0,0,0,0
        for i in range(len(seqs_gold)):
            seq_pred,seq_gold,sent = seqs_pred[i],seqs_gold[i],sents[i]
            seq_pred = np.asarray(seq_pred[1:-1],dtype=str)
            seq_gold = np.asarray(seq_gold[1:],dtype=str)
            kc,unkc,ka,unka = self.count_acc(seq_pred,seq_gold,sent[1:])
            kcc+=kc; unkcc+=unkc; kaa+=ka; unkaa+=unka;
        acc_all = (kcc+unkcc)/(kaa+unkaa)*100 if kaa+unkaa != 0 else 0
        acc_know = kcc/kaa*100 if kaa != 0 else 0
        acc_unk = unkcc/unkaa*100 if unkaa != 0 else 0
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
    ichmm.set_lambda(1,0)
    ichmm.count_estimate(filetrn)
    sents,seqs_gold = readsents(filetst)
    seqs_pred = []
    for i in range(len(sents)):
        sent = sents[i]
        pred = ichmm.viterbi_decode(sent)
        seqs_pred.append(pred)
    ichmm.eval_acc(seqs_pred,seqs_gold,sents)

if __name__ == "__main__":
    main()
