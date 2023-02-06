#!/bin/sh

step=$1
home_dir=../../data/
dataset=tacred
pet_dir=output/sinre/step0
#input
model_dir=${pet_dir}/p3-i0
rel2id_file=${home_dir}/${dataset}/my_rel2id.json
task_name=sinre

#output: pattern file
unlabel_path=$2
save_path=$3
words_file=${pet_dir}/logits_multip.json


python get_dataset_sinre.py \
  --model_dir ${model_dir} \
  --unlabeled_path ${unlabel_path} \
  --task_name ${task_name} \
  --words_file ${words_file} \
  --rel2id_file ${rel2id_file} \
  --save_path ${save_path}


