#!/bin/sh

step=$1

pet_dir=output/multicon/step0
home_dir=../../data/

#input
con2id_file=${home_dir}/step0/ptuning/multi_cls/con2id.json
task_name=multicon
model_path=${pet_dir}/p1-i0
unlabel_path=$2

#output: pattern file
logits_file=${pet_dir}/unlabel_logits.json
save_file=$3

python get_concept.py \
  --model_dir ${model_path} \
  --unlabeled_path ${unlabel_path} \
  --task_name ${task_name} \
  --logits_file ${logits_file} \
  --con2id_file ${con2id_file} \
  --save_path ${save_file}





