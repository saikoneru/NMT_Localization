#!/bin/bash
SRC=$1
TGT=$2
ROOT=".."
DATA=$ROOT/data/wmt_data/raw/
OUT=$ROOT/data/wmt_data/tokenized/

TOKENIZER=$ROOT/mosesdecoder/scripts/tokenizer/tokenizer.perl
MOSES=$ROOT/mosesdecoder/scripts
LC=$ROOT/mosesdecoder/scripts/tokenizer/lowercase.perl
CLEAN=$ROOT/mosesdecoder/scripts/training/clean-corpus-n.perl

mkdir -p $OUT

function mosestokenize(){
        lg=$1
        if [ ! -d "$MOSES" ]; then
		cd $ROOT
                git clone https://github.com/moses-smt/mosesdecoder.git
		cd -
        fi
        $TOKENIZER -threads 8 -l $lg < "$DATA/train.$lg" > "$OUT/train.$lg"
        $TOKENIZER -threads 8 -l $lg < "$DATA/valid.$lg" > "$OUT/valid.$lg"
        $TOKENIZER -threads 8 -l $lg < "$DATA/test.$lg" > "$OUT/test.$lg"
}

mosestokenize $SRC
mosestokenize $TGT


