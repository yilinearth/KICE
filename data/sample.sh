#!/bin/sh

step=0

dataset=tacred
#input
rel2id_path=${dataset}/my_rel2id.json
total_path=${dataset}/my_train.json
#output
mkdir step${step}
mkdir step${step}/RE_model_data
samp_path=step${step}/RE_model_data/train.json
rest_path=step${step}/rt_aft_samp.json

samp_num=86
python sample.py \
  --rel2id_path ${rel2id_path} \
  --total_data_path ${total_path} \
  --samp_data_path ${samp_path} \
  --rest_path ${rest_path}

total_path=${dataset}/my_dev.json
samp_path=step${step}/RE_model_data/dev.json
rest_path=step${step}/rd_aft_samp.json

python sample.py \
  --rel2id_path ${rel2id_path} \
  --total_data_path ${total_path} \
  --samp_data_path ${samp_path} \
  --rest_path ${rest_path}
