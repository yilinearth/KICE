#!/bin/sh
model_name_or_path=roberta-base
dataset_name=tacred
python get_label_word.py \
--model_name_or_path ${model_name_or_path} \
--dataset_name ${dataset_name}