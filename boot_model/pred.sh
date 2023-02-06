home_dir=../data/
dataset=tacred
rel2id_file=${home_dir}/${dataset}/my_rel2id.json

#input
step=$1
old_step=$2
pred_test_file=$3
ckpt_path=ckpt/bert_bag_avg_step${old_step}.pth.tar

max_length=128
topk_per_relation=3 #how many self-inferred rules per relation
batch_size=30

#output:new rules and rest data
result_path=${home_dir}/step${step}/rule_data/can_rules.json
rest_path=${home_dir}/step${step}/rt_aft_pred.json


python3.6 get_pred_bag_bert.py \
  --max_length ${max_length} \
  --topk_per_relation ${topk_per_relation} \
  --batch_size ${batch_size} \
  --rel2id_path ${rel2id_file} \
  --pred_test_file ${pred_test_file} \
  --ckpt ${ckpt_path} \
  --result_path ${result_path} \
  --rest_path ${rest_path}



