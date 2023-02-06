#!/bin/sh


step=$1
old_step=$2

cd data
dir_name=step${step}
mkdir ${dir_name}
cd ${dir_name}
mkdir annotate
mkdir rule_data
mkdir RE_model_data
mkdir active

cd ..
cd ..
#choose most confused data to build new rules
cd boot_model
#bash get_query.sh ${step} ${old_step}


#rule annotating
cd ..
cd annotate
home_dir=../data/
emb_data_path=${home_dir}/step${step}/rest_aft_act.json
emb_rule_path=${home_dir}/step${step}/rule_data/act_rules.json
#bash anno.sh ${step} ${old_step} ${emb_data_path} ${emb_rule_path}

#filter no relation data from annotated data and label them with no relation
cd ..
cd ptuning
cd PT_Fewshot
unlabel_path=../../data/step${step}/annotate/add_data_200_na_100.json
save_path=../../data/step${step}/annotate/add_data_200_fix.json
#bash label_data_boolre.sh ${step} 100 save_fix ${unlabel_path} ${save_path}


#add annotated data and new generated rules to training set
cd ..
cd ..
cd data
add_data=step${step}/annotate/add_data_200_fix.json
old_train=step${old_step}/RE_model_data/train.json
new_train=step${step}/RE_model_data/train_tmp.json
type=simple
#python combine.py \
#  --in_path1 ${old_train} \
#  --in_path2 ${add_data} \
#  --output_path ${new_train} \
#  --type ${type}

old_train=step${step}/RE_model_data/train_tmp.json
add_data=step${step}/active/query_unlabel_rules.json
new_train=step${step}/RE_model_data/train.json
#python combine.py \
#  --in_path1 ${old_train} \
#  --in_path2 ${add_data} \
#  --output_path ${new_train} \
#  --type ${type}


#train the RE model
cd ..
cd boot_model
bash train_self.sh ${step} ${old_step}

