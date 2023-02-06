
home_dir=../data/
dataset=tacred
step=$1
old_step=$2

add_train_file=${home_dir}/step${step}/annotate/add_data_200_fix.json
val_file=${home_dir}/step0/RE_model_data/dev.json
test_file=${home_dir}/${dataset}/my_test.json
rel2id_file=${home_dir}/${dataset}/my_rel2id.json
#output
res_file=${home_dir}/step${step}/RE_model_data/bag_res_200_self.json
ckpt=bert_bag_avg_step${step}
old_ckpt=bert_bag_avg_step${old_step}

python3.6 train_bag_bert_self.py \
--metric mic_f1 \
--batch_size 28 \
--max_epoch 5 \
--max_length 120 \
--seed 42 \
--aggr avg \
--bag_size 2 \
--res_file ${res_file} \
--train_file ${add_train_file} \
--val_file ${val_file} \
--test_file ${test_file} \
--rel2id_file ${rel2id_file} \
--ckpt ${ckpt} \
--old_ckpt ${old_ckpt} \
--calc_loss_weight 0.5 \
--is_focal_loss


