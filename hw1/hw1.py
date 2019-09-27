#!/usr/bin/env python3
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--t", help="print trees")
    parser.add_argument("-M",type=int,help="recursion limit")
    parser.add_argument("gram",type=str,
                        help="grammar file")
    parser.add_argument("num",type=int,default=1,nargs='?',
                        help="num of sentence to generate")
    args = parser.parse_args()
    return args

class gramNode(): #a node of parse tree
    def __init__(self,nonTerm):
        self.nonTerm = nonTerm  
        self.probs = [] #probability to expand a particular rule
        self.rules = [] #children

    def select(self): #select rule to expand with probability
        probs =self.probs/np.sum(self.probs)
        ind = np.random.choice(len(probs),p=probs)
        return self.rules[ind]
    
    def addRule(self,rule,prob):
        self.probs.append(prob)
        self.rules.append(rule)

class treeNode():
    def __init__(self,value):
        self.value = value #could be non-terminal or terminal
        self.children = None

    def setChild(self,child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)

def readGram(gram):
    gram_dict = {}
    with open(gram,'r') as file:
        for line in file:
            #check whitespace and comment
            if line.strip() and line.strip()[0] != "#":
                ck = line.find("#") #check in-line comments
                line = line if ck==-1 else line[:ck]
                splits = line.split() 
                prob = splits[0] 
                lhs = splits[1]
                rhs = splits[2:]
                #if lhs not in gram_dict.keys():
                #    gram_dict[lhs] = [rhs]
                #else:
                #    gram_dict[lhs].append(rhs)
                if lhs not in gram_dict.keys():
                    node = gramNode(lhs)
                    node.addRule(rhs,float(prob))
                    gram_dict[lhs] = node
                else:
                    gram_dict[lhs].addRule(rhs,float(prob))
    assert(gram_dict) #check emptiness
    return gram_dict

def expandGram(gram_dict,node,M=0):
    chosen_rhs = gram_dict[node.value].select() 
#    M+=1 #record recursion depth
    for constituent in chosen_rhs:
        if constituent in gram_dict.keys(): #it is a non-terminal
#            if M < 450:
                new_child = treeNode(constituent)
                node.setChild(new_child) #add child to parent
                expandGram(gram_dict,new_child,M) #depth first
#            else:
#                new_child = treeNode("...") #limiting phrase
#                node.setChild(new_child)
        else:  #it is a terminal
            new_child = treeNode(constituent)
            node.setChild(new_child)
    return M

def inorderTraversal(node,sentence=[]):
    if node.children is not None:
        for child in node.children:
            inorderTraversal(child,sentence)
    else:
        sentence.append(node.value)
    return sentence

#def expandGram(gram_dict,constituent,sentence,M=0):
#    rhsides = [gram_dict[constituent][i][0] for i in range(len(gram_dict[constituent]))]
#    probs = [gram_dict[constituent][i][1] for i in range(len(gram_dict[constituent]))] 
#    probs = np.asarray(probs,dtype=float)
#    print(probs)
#    probs = probs/np.sum(probs)
#    ind = np.random.choice(np.arange(len(probs)),p=probs)
#    #ind = np.random.randint(low=0,high=len(rhsides)) #high is exclusive
#    rhs = rhsides[ind]
#    for constituent in rhs:
#        if constituent in gram_dict.keys(): #it is a non-terminal
#            M += 1
#            if M < 450:
#                expandGram(gram_dict,constituent,sentence,M)
#            else:
#                sentence.append("...")
#        else:  #it is a terminal
#            sentence.append(constituent)
#    return

def genTree(gram_dict):
    root = treeNode("ROOT")
    expandGram(gram_dict,root)
    return root
    #root = treeNode("ROOT")
    #expandGram(gram_dict,"ROOT",root)
    #root = treeNode("ROOT")
    #print((" ").join(sentence))

def main():
    args = get_args()
    gram,num = args.gram,args.num
    gram_dict = readGram(gram)
    for i in range(num):
        root=genTree(gram_dict)
        sentence=inorderTraversal(root)
        print((" ").join(sentence))

if __name__=="__main__":
    main()
