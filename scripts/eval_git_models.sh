#/bin/bash

ROOT=".."
src=en
tgt=de
fairseq=$ROOT/fairseq/
seed=2
version=v1
input_dir=$ROOT/data/gitdata_parsed_$tgt/$version/
output_dir=$fairseq/checkpoints/$version/git/wmt_finetune/
cxt_types=(none domain filename neighbour)
#cxt_types=(none)
model_type="best"
MOSES=$ROOT/mosesdecoder/scripts
charpy=$ROOT/CharacTER/CharacTER.py
avg_checkpoints=5
eval_set=rnd
#gnome_eval=gnome_final/v2/cxt_data/

if [ ! -d "$MOSES" ]; then
	cd $ROOT
	git clone https://github.com/moses-smt/mosesdecoder.git
	cd -
fi

if [ ! -f "$charpy" ]; then
	cd $ROOT
	git clone https://github.com/rwth-i6/CharacTER
	cd -
fi



for cxt_type in "${cxt_types[@]}"; do
	for dir in "${src}-${tgt}" "${tgt}-${src}"; do
		IFS='-'
		read -ra src_tgt <<< "$dir"
		src_lang=${src_tgt[0]}
		tgt_lang=${src_tgt[1]}
		save_dir=$output_dir/${cxt_type}_${seed}/$dir/
		hyp_file=$save_dir/hyp.$eval_set.$tgt_lang
		echo "Scores for model $save_dir and direction $dir and model type $model_type"
		IFS=''
		if [ $model_type = 'avg' ]; then
			python $fairseq/scripts/average_checkpoints.py --inputs $save_dir --output $save_dir/checkpoint_avg.pt --num-epoch-checkpoints $avg_checkpoints
		fi
		#cp $fairseq/data-bin/$version/git/wmt/dict.* $fairseq/data-bin/$version/gnome/rnd/wmt_finetune/$cxt_type/$dir
		
		if [ $eval_set = 'rnd' ]; then
			input_file=$input_dir/${cxt_type}/rnd/$dir/test.$src_lang
			ref_file=$input_dir/none/rnd/$dir/raw/test.$tgt_lang
    fi
    if [ $eval_set == 'unknown' ]; then
			input_file=$input_dir/${cxt_type}/$dir/test.$src_lang
			ref_file=$input_dir/none/$dir/raw/test.$tgt_lang
		fi
    if [ $eval_set == 'gnome_unknown' ]; then
			input_file=$gnome_eval/${cxt_type}_bpe/$dir/test.$src_lang
			ref_file=$gnome_eval/${cxt_type}/raw/$dir/test.$tgt_lang
		fi
    if [ $eval_set == 'gnome_rnd' ]; then
			input_file=$gnome_eval/${cxt_type}_bpe/$dir/rnd/test.$src_lang
			ref_file=$gnome_eval/${cxt_type}/rnd/raw/$dir/test.$tgt_lang
		fi
		fairseq-interactive $fairseq/data-bin/$version/git/wmt_finetune/${cxt_type}/$dir --path ${save_dir}/checkpoint_${model_type}.pt --beam 5 --source-lang $src_lang --target-lang $tgt_lang --input $input_file --buffer-size 128 --batch-size 32 --remove-bpe=subword_nmt   > $output_dir/${cxt_type}_${seed}/$dir/pred.log.$model_type.$eval_set
		wait
		grep ^H $output_dir/${cxt_type}_${seed}/$dir/pred.log.$model_type.$eval_set | cut -f3 > $hyp_file.tok
		$MOSES/tokenizer/detokenizer.perl -l $tgt_lang < $hyp_file.tok > $hyp_file
		cat $hyp_file | sacrebleu $ref_file -m bleu -b -w 4 -sh
		python CharacTER/CharacTER.py -r $ref_file -o $hyp_file
		echo "##############################################################"
		echo "##############################################################"
	done
done

wait


