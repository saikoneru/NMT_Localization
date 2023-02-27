#!/bin/bash

src=en
tgt=de
version=v1
ROOT=".."
bpe_data=$ROOT/data/wmt_data/bpe/


for dir in "${src}-${tgt}" "${tgt}-${src}"; do
  IFS='-'
  read -ra src_tgt <<< "$dir"
  src_lang=${src_tgt[0]}
  tgt_lang=${src_tgt[1]}
  IFS=''
  data_bin=$ROOT/fairseq/data-bin/$version/git/wmt/${src_lang}-${tgt_lang}/
  mkdir -p $data_bin
  fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
    --trainpref $bpe_data/train --validpref $bpe_data/valid --testpref $bpe_data/test \
    --destdir $data_bin --joined-dictionary
done
