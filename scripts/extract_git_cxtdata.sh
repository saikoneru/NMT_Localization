#!/bin/bash

ROOT=".."
version=v1
src=en
tgt=de
out_dir=$ROOT/data/gitdata_parsed_$tgt/v1/
data=$ROOT/data/gitdata_parsed_$tgt/data.csv
#gnome_eval=$ROOT/gnome_final/v2/cxt_data/none_bpe/full_eval/
bpe_data_path=$ROOT/data/gitdata_parsed_$tgt/
train_split=0.996
neighbour_size=1
num_cluster=30
SUBWORD=$ROOT/subword-nmt/subword_nmt
wmt_data=$ROOT/wmt_data/bpe/
CODES=$wmt_data/code
fairseq=$ROOT/fairseq/
cxt_types=(neighbour none domain filename)
dict_path=$fairseq/data-bin/$version/git/dict/

mkdir -p $dict_path

for cxt_type in "${cxt_types[@]}"; do
	python3 create_git_cxtdata.py --data_path $data  --bpe_data_path $bpe_data_path --output_path $out_dir/$cxt_type --src_lang $src --tgt_lang $tgt --append_src $cxt_type --neighbour_size $neighbour_size --train_split $train_split --num_cluster $num_cluster 
done


echo "Finished splitting the data based on contexts"
rm -rf tmp/
mkdir -p tmp
for lg in $src $tgt; do
  cat $wmt_data/train.$lg   > tmp/train.$lg
  head $out_dir/neighbour/$src-$tgt/train.$lg >> tmp/train.$lg
done
for split in train valid test; do
  for lg in $src $tgt; do
  cat $out_dir/domain/$src-$tgt/$split.$lg >> tmp/train.$lg  
  cat $out_dir/filename/$src-$tgt/$split.$lg >> tmp/train.$lg  
  done
done
echo "Generating dictionary for fairseq"
fairseq-preprocess --source-lang $src --target-lang $tgt  --trainpref tmp/train  --destdir $dict_path  --joined-dictionary --dict-only --workers 10


for dir in "${src}-${tgt}" "${tgt}-${src}"; do
  for cxt_type in "${cxt_types[@]}"; do
		IFS='-'
		read -ra src_tgt <<< "$dir"
		src_lang=${src_tgt[0]}
		tgt_lang=${src_tgt[1]}
    train_bpe_dir=$out_dir/${cxt_type}/$dir/
		eval_bpe_dir=$out_dir/${cxt_type}/rnd/$dir/
		IFS=''
    echo "Processing Context type $cxt_type for direction $dir "
		fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang  --trainpref $train_bpe_dir/train --validpref $eval_bpe_dir/valid --testpref $eval_bpe_dir/test  --destdir $fairseq/data-bin/$version/git/wmt_finetune/$cxt_type/$dir/  --srcdict $dict_path/dict.$src_lang.txt  --tgtdict $dict_path/dict.$tgt_lang.txt --workers 10
		cp $dict_path/dict* $fairseq/data-bin/$version/git/wmt_finetune/$cxt_type/$dir/
	done
  echo "Processing WMT for direction $dir"
  fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang  --trainpref $wmt_data/train --validpref $wmt_data/valid --testpref $wmt_data/test  --destdir $fairseq/data-bin/$version/git/wmt/$dir/  --srcdict $dict_path/dict.$src_lang.txt  --tgtdict $dict_path/dict.$tgt_lang.txt --workers 10
  cp $dict_path/dict* $fairseq/data-bin/$version/git/wmt/$dir/
done
