#!/usr/bin/env python3
import argparse
import numpy as np
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",type=str,
                        help="MODE of Operation")
    parser.add_argument("gram",type=str,
                        help="grammar file")
    parser.add_argument("sent",type=str,
                        help="sentence file")
    args = parser.parse_args()
    return args

class gramNode(): #a node of parse tree
    def __init__(self,lhs,rhs,prob):
        self.lhs = lhs
        self.rhs = rhs
        self.prob = prob

class treeNode():
    def __init__(self,gram_node):
        self.node = gram_node
        self.children = None
        self.parents = None

    def set_children(self,children):
        if self.children is None:
            self.children = [children]
        else:
            self.children = self.children.append(children)

    def set_parent(self,parent):
        if self.parents is None:
            self.parents = [parent]
        else:
            self.parents = self.parents.append(parent)


def read_grammar(gram):
    grammar = {}
    with open(gram,'r') as gram_file:
        for line in gram_file:
            parts = line.strip().split()
            #get rhs terminals or non-terminals
            key = (parts[2],parts[3]) if len(parts) == 4 else (parts[2],None)
            #store prob and lhs of the rule,0 is prob, 1 is lhs
            if key in grammar.keys():
                grammar[key].append((parts[0],parts[1]))
            else:
                grammar[key] = [(parts[0],parts[1])]
    return grammar



def main():
    args = get_args()
    mode,gram,sent = args.mode,args.gram,args.sent
    grammar = read_grammar(gram)
    recognizer, parse_weight, total_weight = [],[],[]
    with open(sent,'r') as sent_file:
        for line in sent_file:
            status = "pass"
            words = line.strip().split()
            lw = len(words)
            table = np.zeros((lw,lw)).tolist()
            #loop words along superdiagnal
            for j in range(1,lw+1):
                word = words[j]
                if (word,None) in grammar.keys():
                    treeNodes = []
                    #store each rule that generates the word; rule[1]=lhs, rule[0]=prob
                    for rule in grammar[word]:
                        gram_node = gramNode(rule[1],word,rule[0])
                        new_treeNode = treeNode(gram_node)
                        treeNodes.append(new_treeNode)
                    table[j-1][j] = treeNodes
                else:
                    status = "fail"; break;
                #loop rows at current column
                for i in range(j-2,-1,-1):
                    # loop possible parse positions bw. i+1 and j-1
                    for k in range(i+1,j):
                        treeNodes = []
                        nT1s,nT2s = table[i][k],table[k][j]
                        for idx1 in range(len(nT1s)):
                            nT1 = nT1s[idx1].node.lhs
                            for idx2 in range(len(nT2s)):
                                nT2 = nT2s[idx2].node.lhs
                                if (nT1,nT2) in grammar.keys():
                                    for rule in grammar[(nT1,nT2)]:
                                        gram_node = gramNode(rule[1],word,rule[0])
                                        new_treeNode = treeNode(gram_node)
                                        new_treeNode.set_children((nT1s[idx1],nT2s[idx2]))
                                        nT1s[idx1].set_parent(new_treeNode) #TODO: is parent necessary?
                                        nT2s[idx2].set_parent(new_treeNode)
                                        treeNodes.append(new_treeNode)
                    table[i][j] = treeNodes








if __name__ == "__main__":
    main()




