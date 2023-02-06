#!/bin/sh
#ptuning
step=0
cd data
bash ptuning_pre.sh
cd ..

cd ptuning
cd PT_Fewshot
# ptuning plm for concept pattern extraction
bash scripts/multicon_pt_few_shot.sh

# ptuning plm for relation pattern extraction
bash scripts/sinre_pt_few_shot.sh

# ptuning plm to classify na relation
bash scripts/boolre_pt_few_shot.sh

#filter out partial no relation data in unlabeled dataset
unlabel_path=../../data/step${step}/rt_aft_samp.json
save_path=../../data/step${step}/rt_aft_samp_valid.json
bash label_data_boolre.sh ${step} 25000 save_valid ${unlabel_path} ${save_path}

#extract the concept pattern of initial training set and unlabeled set
home_dir=../../data/
unlabel_path=${home_dir}/step${step}/RE_model_data/train.json
save_file=${home_dir}/step${step}/RE_model_data/train_con.json
bash label_data_multicon.sh ${step} ${unlabel_path} ${save_file}

unlabel_path=${home_dir}/step${step}/rt_aft_samp_valid.json
save_path=${home_dir}/step${step}/rt_aft_samp_valid_con.json
bash label_data_multicon.sh ${step} ${unlabel_path} ${save_path}

#extract the relation pattern of initial training set and unlabeled set
unlabel_path=${home_dir}/step${step}/RE_model_data/train_con.json
save_path=${home_dir}/step${step}/RE_model_data/train_conrel.json
bash label_data_sinre.sh ${step} ${unlabel_path} ${save_path}

unlabel_path=${home_dir}/step${step}/rt_aft_samp_valid_con.json
save_path=${home_dir}/step${step}/rt_aft_samp_valid_conrel.json
bash label_data_sinre.sh ${step} ${unlabel_path} ${save_path}


#get pattern emb for training set and unlabeled set
cd ..
cd ..
cd annotate
home_dir=../data/
data_path=${home_dir}/step${step}/RE_model_data/train_conrel.json
emb_data_path=${home_dir}/step${step}/RE_model_data/train_emb.json
bash emb.sh ${data_path} ${emb_data_path}

data_path=${home_dir}/step${step}/rt_aft_samp_valid_conrel.json
emb_data_path=${home_dir}/step${step}/rt_aft_samp_valid_emb.json
bash emb.sh ${data_path} ${emb_data_path}

#train a RE model for step 0
cd ..
cd boot_model
bash train.sh ${step}