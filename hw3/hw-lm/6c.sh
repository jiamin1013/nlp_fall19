#!/bin/bash
[[ -f result.txt ]] && rm result.txt
for c in 0.05 1 20 50; do
  for dim in 10 20 40; do
    if [ ! -f loglinear${c}_chars-$dim.txt_en.1K.model ] || [ ! -f loglinear${c}_chars-$dim.txt_sp.1K.model ]; then
      python3 textcat.py TRAIN loglinear$c lexicons/chars-$dim.txt english_spanish/train/en.1K english_spanish/train/sp.1K
    fi
#    if [ ! -f loglinear${c}_chars-$dim.txt_sp.1K.model ]; then
#      python3 textcat.py TRAIN loglinear$c lexicons/chars-$dim.txt english_spanish/train/sp.1K 
#    fi
    python3 textcat.py TEST loglinear$c lexicons/chars-$dim.txt english_spanish/train/en.1K english_spanish/train/sp.1K 0.7 english_spanish/dev/*/*/* > rr
    echo "C=$c, dim=$dim:" >> result.txt
    python3 calAcc.py >> result.txt
  done
done
