#!/bin/sh

step=$1
save_num=$2
type=$3
unlabel_path=$4

dataset=tacred
#input
pet_dir=output/boolre/step0
home_dir=../../data/
model_path=${pet_dir}
rel2id_file=${home_dir}/${dataset}/my_rel2id.json
task_name=boolre
#output
logits_file=${pet_dir}/unlabel_logits.json
save_path=$5



python get_dataset.py \
  --model_dir ${model_path} \
  --unlabeled_path ${unlabel_path} \
  --task_name ${task_name} \
  --logits_file ${logits_file} \
  --rel2id_file ${rel2id_file} \
  --save_path ${save_path} \
  --save_num ${save_num} \
  --type ${type}