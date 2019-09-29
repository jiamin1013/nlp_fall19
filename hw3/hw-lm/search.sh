#!/bin/bash

for lambda in -3, -2, -1, 0, 1, 2, 3; do
  lamda=`python -c "print(10**$lamda)"` 
  #Change fileprob later
  python3 fileprob.py TRAIN "add$lamda" lexicons/words-10.txt english_spanish/train/en.1K
  python3 fileprob.py TEST "add$lamda" lexicons/words-10.txt english_spanish/train/en.1K english_spanish/dev/english/*/*
done

