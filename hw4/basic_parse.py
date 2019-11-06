#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.special import logsumexp
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

class backNode():
    def __init__(self,k,nT1,nT2):
        self.k = k
        self.nT1 = nT1
        self.nT2 = nT2

class cell():
    def __init__(self,ruleProbs,total_weights,total_weight):
        self.ruleProbs = ruleProbs
        self.tws = total_weights
        self.tw_best = total_weight

def read_grammar(gram):
    grammar = {}
    with open(gram,'r') as gram_file:
        for line in gram_file:
            parts = line.strip().split()
            #get rhs terminals or non-terminals
            key = (parts[2],parts[3]) if len(parts) == 4 else (parts[2],None)
            #store prob and lhs of the rule,0 is prob, 1 is lhs
            if key in grammar.keys():
                grammar[key].append((np.log(float(parts[0])),parts[1])) #log prob
            else:
                grammar[key] = [(np.log(float(parts[0])),parts[1])]
    return grammar

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
            back = np.zeros((lw+1,lw+1)).tolist()
            #loop words along superdiagnal
            for j in range(1,lw+1):
                word = words[j-1]
                if (word,None) in grammar.keys():
                    ruleProbs = {}
                    #store each rule that generates the word; rule[1]=lhs, rule[0]=prob
                    for rule in grammar[(word,None)]:
                        ruleProbs[rule[1]] = rule[0]
                    tw_best = np.max(ruleProbs.values())
                    table[j-1][j] = cell(ruleProbs,ruleProbs,tw_best)
                    back[j-1][j] = word
                else:
                    break; #if a terminal non-exsists, no parse exsits, terminate
                #loop rows at current column
                for i in range(j-2,-1,-1):
                    ruleProbs,tw,backmap = {},{},{}
                    # loop possible parse positions bw. i+1 and j-1
                    for k in range(i+1,j):
                            for nT1 in table[i][k].ruleProbs.keys():
                                for nT2 in table[k][j].ruleProbs.keys():
                                    if (nT1,nT2) in grammar.keys():
                                        for rule in grammar[(nT1,nT2)]:
                                            p = rule[0]+table[i][k].ruleProbs[nT1]+table[k][j].ruleProbs[nT2]
                                            newTw = rule[0]+table[i][k].tws[nT1]+table[k][j].tws[nT2]
                                            if rule[1] in ruleProbs.keys():
                                                tw[rule[1]] = logsumexp([tw[rule[1]],newTw])
                                                if ruleProbs[rule[1]] < p:
                                                    ruleProbs[rule[1]] = p
                                                    backmap[rule[1]] = backNode(k,nT1,nT2)
                                            else:
                                                ruleProbs[rule[1]] = p
                                                tw[rule[1]] = newTw
                                                backmap[rule[1]] = backNode(k,nT1,nT2)
                    best_total = np.max(list(tw.values())) if len(tw) != 0 else None
                    table[i][j] = cell(ruleProbs,tw,best_total)
                    back[i][j] = backmap
            if table[0][lw] != 0:
                if "ROOT" in table[0][lw].ruleProbs.keys():
                    recognizer.append("True")
                    tw_best = -np.log2(np.exp(1))*table[0][lw].ruleProbs["ROOT"]
                    parse =  buildTree(back,0,lw,"ROOT",["(ROOT "])
                    best_parse.append("{0:.3f}\t".format(tw_best)+"".join(parse).strip())
                    tolw = -np.log2(np.exp(1))*logsumexp(list(table[0][lw].tws.values()))
                    total_weight.append("{0:.3f}".format(tolw))
                else:
                    recognizer.append("False")
                    best_parse.append("-\tNOPARSE")
                    total_weight.append("-")
            else:
                    recognizer.append("False")
                    best_parse.append("-\tNOPARSE")
                    total_weight.append("-")
        if mode == "RECOGNIZER":
            print("\n".join(recognizer))
        elif mode == "BEST-PARSE":
            print("\n".join(best_parse))
        elif mode == "TOTAL-WEIGHT":
            print("\n".join(total_weight))


def buildTree(back,row,col,nT,parse):
    if type(back[row][col]) != str:
        backNode = back[row][col][nT]
        parse.append("({0} ".format(backNode.nT1))
        row1,col1 = row,backNode.k
        parse = buildTree(back,row1,col1,backNode.nT1,parse)
        row2,col2 = backNode.k,col
        parse.append(" ({0} ".format(backNode.nT2))
        parse = buildTree(back,row2,col2,backNode.nT2,parse)
        parse.append(")")
    else:
        parse.append("{0})".format(back[row][col]))
    return parse



#def main():
#    args = get_args()
#    mode,gram,sent = args.mode,args.gram,args.sent
#    grammar = read_grammar(gram)
#    recognizer, best_parse, total_weight = [],[],[]
#    with open(sent,'r') as sent_file:
#        for line in sent_file:
#            words = line.strip().split()
#            lw = len(words)
#            table = np.zeros((lw+1,lw+1)).tolist()
#            #loop words along superdiagnal
#            for j in range(1,lw+1):
#                word = words[j-1]
#                if (word,None) in grammar.keys():
#                    treeNodes = []
#                    #store each rule that generates the word; rule[1]=lhs, rule[0]=prob
#                    for rule in grammar[(word,None)]:
#                        gram_node = gramNode(rule[1],word,rule[0])
#                        new_treeNode = treeNode(gram_node)
#                        treeNodes.append(new_treeNode)
#                    table[j-1][j] = treeNodes
#                else:
#                    recognizer.append("False")
#                    best_parse.append("-\tNOPARSE")
#                    total_weight.append("-")
#                    break; #if a terminal non-exsists, no parse exsits, terminate
#                #loop rows at current column
#                for i in range(j-2,-1,-1):
#                    treeNodes = []
#                    # loop possible parse positions bw. i+1 and j-1
#                    for k in range(i+1,j):
#                        nT1s,nT2s = table[i][k],table[k][j]
#                        for idx1 in range(len(nT1s)):
#                            nT1 = nT1s[idx1].node.lhs
#                            for idx2 in range(len(nT2s)):
#                                nT2 = nT2s[idx2].node.lhs
#                                if (nT1,nT2) in grammar.keys():
#                                    for rule in grammar[(nT1,nT2)]:
#                                        gram_node = gramNode(rule[1],None,rule[0]) #No need to store rhs, because it's children
#                                        new_treeNode = treeNode(gram_node)
#                                        new_treeNode.set_children((nT1s[idx1],nT2s[idx2]))
#                                        #nT1s[idx1].set_parent(new_treeNode) #TODO: is parent necessary?
#                                        #nT2s[idx2].set_parent(new_treeNode)
#                                        treeNodes.append(new_treeNode)
#                    table[i][j] = treeNodes
#            if table[0][lw] != 0:
#                parse = printParse(table[0][lw][0],[])
#                print("".join(parse).strip())
#            else:
#                print("NOPARSE")


if __name__ == "__main__":
    main()




