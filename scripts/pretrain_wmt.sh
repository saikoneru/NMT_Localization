#!/bin/bash

src_lang=en
tgt_lang=de
version=v1
ROOT=".."
fairseq=$ROOT/fairseq
data_path=$fairseq/data-bin/$version/git/wmt/$src_lang-$tgt_lang/
checkpoint=$fairseq/checkpoints/$version/git/$src_lang-$tgt_lang/

mkdir -p $checkpoint

fairseq-train $data_path \
  --arch transformer \
  --dropout 0.2\
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)'  \
  --lr 0.0001 --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --max-tokens 3000 \
  --save-dir $checkpoint \
  --patience 5 \
  --encoder-embed-dim 512 \
  --decoder-embed-dim 512 \
  --encoder-layers 6 \
  --decoder-layers 6 \
  --encoder-attention-heads 4 \
  --decoder-attention-heads 4 \
  --validate-interval-updates 10000 \
  --no-epoch-checkpoints --fp16
