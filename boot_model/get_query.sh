
home_dir=../data/
dataset=tacred

step=$1
old_step=$2
#input
rel2id_path=${home_dir}${dataset}/my_rel2id.json
total_path=${home_dir}/step${old_step}/rt_aft_pred.json
ckpt_path=ckpt/bert_bag_avg_step${old_step}.pth.tar
boot_path=${home_dir}/step${old_step}/rule_data/boot_rules.json
#output
query_unlabel_path=${home_dir}/step${step}/active/query_unlabel_rules.json
query_boot_path=${home_dir}/step${step}/active/query_boot_rules.json
rest_query_path=${home_dir}/step${step}/rest_aft_act.json
conf_path=${home_dir}/step${step}/active/confuse_rule.json


max_length=120
topk=40
query_rule_num=20

batch_size=28
mask_num=3

python3.6 get_active_query.py \
  --max_length ${max_length} \
  --topk ${topk} \
  --query_rule_num ${query_rule_num} \
  --batch_size ${batch_size} \
  --rel2id_path ${rel2id_path} \
  --unlabel_rule_path ${total_path} \
  --boot_rule_path ${boot_path} \
  --conf_rule_path ${conf_path} \
  --ckpt ${ckpt_path} \
  --query_path ${query_unlabel_path} \
  --query_boot_path ${query_boot_path} \
  --rest_path ${rest_query_path} \
  --mask_num ${mask_num} \
  --isSelfFix

#add new rules to rules set
cd ..
cd data
new_rule_path=step${step}/rule_data/act_rules.json
type=add_query_rules
python combine.py \
  --old_rule_file ${boot_path} \
  --query_unlabel_file ${query_unlabel_path} \
  --query_boot_file ${query_boot_path} \
  --new_rule_file ${new_rule_path} \
  --type ${type}