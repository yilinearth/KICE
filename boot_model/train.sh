
home_dir=../data/
dataset=tacred

test_file=${home_dir}/${dataset}/my_test.json
rel2id_file=${home_dir}/${dataset}/my_rel2id.json

step=0
old_step=0
train_file=${home_dir}/step${step}/RE_model_data/train.json
val_file=${home_dir}/step0/RE_model_data/dev.json
#output
#eval result file
res_file=${home_dir}/step${step}/RE_model_data/bag_res.json
#checkpoint file
ckpt=bert_bag_avg_step${step}
mkdir ckpt

python3.6 train_bag_bert.py \
--metric mic_f1 \
--batch_size 25 \
--max_epoch 5 \
--max_length 100 \
--seed 42 \
--aggr avg \
--bag_size 4 \
--res_file ${res_file} \
--train_file ${train_file} \
--val_file ${val_file} \
--test_file ${test_file} \
--rel2id_file ${rel2id_file} \
--ckpt ${ckpt} \
--is_focal_loss
