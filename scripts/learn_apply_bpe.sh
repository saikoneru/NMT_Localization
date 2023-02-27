#!/bin/bash

$ROOT=".."
SRC=de
TGT=en

DATA=$ROOT/data/wmt_data/tokenized/
OUTPATH=$ROOT/data/wmt_data/bpe/
SUBWORD=$ROOT/subword-nmt/subword_nmt

TRAIN=$OUTPATH/train.$SRC-$TGT
CODES=$OUTPATH/code
CODE_OPS=30000

mkdir -p $OUTPATH

if [ ! -d "$SUBWORD" ]; then
	git clone https://github.com/rsennrich/subword-nmt.git
fi

for l in $SRC $TGT; do
	cat $DATA/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $SUBWORD/learn_joint_bpe_and_vocab.py --input  $TRAIN -s $CODE_OPS -o $CODES --write-vocabulary  $OUTPATH/vocab.all

for L in $SRC $TGT; do
	for f in train.$L valid.$L test.$L; do 
		echo "apply_bpe.py to ${f}..."
		python $SUBWORD/apply_bpe.py -c $CODES < $DATA/$f > $OUTPATH/$f
	done
done

rm $TRAIN
