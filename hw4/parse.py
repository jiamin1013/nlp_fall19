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

    def set_children(self,children):
        self.children = children

class cell():
    def __init__(self,treeNodes,best_parse):
        self.parses = treeNodes
        self.best_parse = best_parse

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

def printParse(tNode, parse):
    parse.append(" ({0} ".format(tNode.node.lhs))
    if tNode.children is not None:
        cNode = tNode.children
        parse = printParse(cNode[0],parse)
        parse = printParse(cNode[1],parse)
        parse.append(")")
    else:
        parse.append("{0})".format(tNode.node.rhs))
    return parse

def main():
    args = get_args()
    mode,gram,sent = args.mode,args.gram,args.sent
    grammar = read_grammar(gram)
    recognizer, best_parse, total_weight = [],[],[]
    with open(sent,'r') as sent_file:
        for line in sent_file:
            words = line.strip().split()
            lw = len(words)
            table = np.zeros((lw+1,lw+1)).tolist()
            #loop words along superdiagnal
            for j in range(1,lw+1):
                word = words[j-1]
                if (word,None) in grammar.keys():
                    treeNodes = []
                    #store each rule that generates the word; rule[1]=lhs, rule[0]=prob
                    for rule in grammar[(word,None)]:
                        gram_node = gramNode(rule[1],word,rule[0])
                        new_treeNode = treeNode(gram_node)
                        treeNodes.append(new_treeNode)
                    table[j-1][j] = treeNodes
                else:
                    recognizer.append("False")
                    best_parse.append("-\tNOPARSE")
                    total_weight.append("-")
                    break; #if a terminal non-exsists, no parse exsits, terminate
                #loop rows at current column
                for i in range(j-2,-1,-1):
                    treeNodes = []
                    # loop possible parse positions bw. i+1 and j-1
                    for k in range(i+1,j):
                        nT1s,nT2s = table[i][k],table[k][j]
                        for idx1 in range(len(nT1s)):
                            nT1 = nT1s[idx1].node.lhs
                            for idx2 in range(len(nT2s)):
                                nT2 = nT2s[idx2].node.lhs
                                if (nT1,nT2) in grammar.keys():
                                    for rule in grammar[(nT1,nT2)]:
                                        gram_node = gramNode(rule[1],None,rule[0]) #No need to store rhs, because it's children
                                        new_treeNode = treeNode(gram_node)
                                        new_treeNode.set_children((nT1s[idx1],nT2s[idx2]))
                                        #nT1s[idx1].set_parent(new_treeNode) #TODO: is parent necessary?
                                        #nT2s[idx2].set_parent(new_treeNode)
                                        treeNodes.append(new_treeNode)
                    table[i][j] = treeNodes
            if table[0][lw] != 0:
                parse = printParse(table[0][lw][0],[])
                print("".join(parse).strip())
            else:
                print("NOPARSE")

def tablePrint(table):
    for row in table:
        print(row)



if __name__ == "__main__":
    main()




