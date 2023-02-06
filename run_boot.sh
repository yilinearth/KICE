#!/bin/sh
home_dir=../data/

step=$1
old_step=$2

cd data
dir_name=step${step}
mkdir ${dir_name}
cd ${dir_name}
mkdir annotate
mkdir rule_data
mkdir RE_model_data

#apply learned model on unlabeled set to get new rules
cd ..
cd ..
cd boot_model
#if old step = 0
pred_test_file=../data/step${old_step}/rt_aft_samp_valid_emb.json
#else
#pred_test_file=../data/step${old_step}/rt_aft_anno.json
bash pred.sh ${step} ${old_step} ${pred_test_file}


#add new rules to rules set
cd ..
cd data
#if old step = 0
old_rule_path=${home_dir}/step${old_step}/RE_model_data/train_emb.json
#if old_step != 0
#old_rule_path=${home_dir}/step${old_step}/rule_data/act_rules.json
add_rule_path=${home_dir}/step${step}/rule_data/can_rules.json
new_rule_path=${home_dir}/step${step}/rule_data/boot_rules.json
type=no_simple
python combine.py \
  --in_path1 ${old_rule_path} \
  --in_path2 ${add_rule_path} \
  --output_path ${new_rule_path} \
  --type ${type}

#rule annotation
cd ..
cd annotate
emb_data_path=${home_dir}/step${step}/rt_aft_pred.json
emb_rule_path=${home_dir}/step${step}/rule_data/boot_rules.json
bash anno.sh ${step} ${old_step} ${emb_data_path} ${emb_rule_path}

#filter no relation data and label them with no relation
cd ..
cd ptuning
cd PT_Fewshot
unlabel_path=../../data/step${step}/annotate/add_data_200_na_100.json
save_path=../../data/step${step}/annotate/add_data_200_fix.json
bash label_data_boolre.sh ${step} 100 save_fix ${unlabel_path} ${save_path}

#add annotated data and new generated rules to training set
cd ..
cd ..
cd data
old_train=step${old_step}/RE_model_data/train.json
add_data=step${step}/annotate/add_data_200_fix.json
new_train=step${step}/RE_model_data/train_tmp.json
type=simple
python combine.py \
  --in_path1 ${old_train} \
  --in_path2 ${add_data} \
  --output_path ${new_train} \
  --type ${type}

old_train=step${step}/RE_model_data/train_tmp.json
add_data=step${step}/rule_data/can_rules.json
new_train=step${step}/RE_model_data/train.json
python combine.py \
  --in_path1 ${old_train} \
  --in_path2 ${add_data} \
  --output_path ${new_train} \
  --type ${type}


#训练模型
cd ..
cd boot_model
bash train_self.sh ${step} ${old_step}

