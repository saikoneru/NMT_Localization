#!/bin/bash

src=en
tgt=de
ROOT=../
fairseq=$ROOT/fairseq/
bpe_dir=$ROOT/data/wmt_data/bpe/
raw_dir=/$ROOT/data/wmt_data/raw/
version=v1

data_dir=$fairseq/data-bin/$version/git/wmt/$src-$tgt/
src_file=$bpe_dir/test.$src

model_dir=checkpoints/$version/git/$src-$tgt/
model_path="$model_dir/checkpoint_best.pt"

out_dir=$model_dir/score
hyp_file=$out_dir/hyp.$tgt

ref_file_orig=$raw_dir/test.$tgt
MOSES=$ROOT/mosesdecoder/scripts
charpy=$ROOT/CharacTER/CharacTER.py
ref_file=$out_dir/ref.$tgt

mkdir -p $out_dir
cp $ref_file_orig $ref_file

fairseq-interactive $data_dir --path $model_path --beam 5 --source-lang $src --target-lang $tgt --input $src_file --buffer-size 128 --batch-size 32 --remove-bpe=subword_nmt > $out_dir/pred.log
wait


grep ^H $out_dir/pred.log | cut -f3 > $hyp_file.tok
$MOSES/tokenizer/detokenizer.perl -l $tgt < $hyp_file.tok > $hyp_file

cat $hyp_file | sacrebleu $ref_file -m bleu -b -w 4
python $charpy -r $ref_file -o $hyp_file

