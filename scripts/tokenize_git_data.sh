#!/bin/bash

SRC=en
TGT=de
ROOT=".."
DATA=$ROOT/data/gitdata_parsed_$TGT/
OUT=$DATA
CODES=$ROOT/data/wmt_data/bpe/code
TOKENIZER=$ROOT/mosesdecoder/scripts/tokenizer/tokenizer.perl
MOSES=$ROOT/mosesdecoder/scripts
SUBWORD=$ROOT/subword-nmt/subword_nmt

for lg in $SRC $TGT; do
	$TOKENIZER -threads 8 -l $lg < "$DATA/full.$lg" > "$OUT/full.tok.$lg"
	python $SUBWORD/apply_bpe.py -c $CODES < $DATA/full.tok.$lg > $OUT/full.tok.$lg.bpe
done
$TOKENIZER -threads 8 -l "en" < "$DATA/full.cxt" > "$OUT/full.tok.cxt"
python $SUBWORD/apply_bpe.py -c $CODES < $DATA/full.tok.cxt > $OUT/full.tok.cxt.bpe

