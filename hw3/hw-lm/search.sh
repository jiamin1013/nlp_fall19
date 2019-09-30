#!/bin/bash

#for lambda in -3 -2 -1 0 1 2 3; do
#  lambda=`python -c "import math; print(math.pow(10,$lambda))"` 
#  model="add$lambda"
#  #run modified fileprob
#  #python3 fileprob_.py TRAIN $model lexicons/words-10.txt english_spanish/train/en.1K
#  #python3 fileprob_.py TEST $model lexicons/words-10.txt english_spanish/train/en.1K english_spanish/dev/english/*/* 
#  python3 fileprob_.py TRAIN $model lexicons/words-10.txt english_spanish/train/sp.1K
#  python3 fileprob_.py TEST $model lexicons/words-10.txt english_spanish/train/sp.1K english_spanish/dev/spanish/*/* 
#done

for lambda in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
  #lambda=`python -c "import math; print(math.pow(10,$lambda))"` 
  model="add$lambda"
  #run modified fileprob
  #python3 fileprob_.py TRAIN $model lexicons/words-10.txt english_spanish/train/en.1K
  #python3 fileprob_.py TEST $model lexicons/words-10.txt english_spanish/train/en.1K english_spanish/dev/english/*/* 
  python3 fileprob_.py TRAIN $model lexicons/words-10.txt english_spanish/train/sp.1K
  python3 fileprob_.py TEST $model lexicons/words-10.txt english_spanish/train/sp.1K english_spanish/dev/spanish/*/* 
done

