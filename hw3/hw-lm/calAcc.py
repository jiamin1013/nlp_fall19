#!/usr/bin/env python3
import numpy as np
err=0
file=0
with open("rr",'r') as infile:
    for line in infile:
        if len(line.strip().split()) == 2:
            f1 = line.strip().split()[0].split('.')[0]
            f2 = line.strip().split()[1]
            if f1 != f2.split('/')[-1].split('.')[0]:
                err +=1
            file+=1
print("Accuracy: {0:.2f}".format((file-err)/file))
