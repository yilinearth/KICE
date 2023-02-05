
home_dir=../data/
dataset=tacred

step=$1
old_step=$2
rel2id_file=${home_dir}/${dataset}/my_rel2id.json
emb_data_path=$3
emb_rule_path=$4

mask_num=3
top_num=200 #pick topk data with highest labeling confidence
type=knn_best
threshold=0.97
#output
tmp_path=${home_dir}/step${step}/annotate/anno_tmp.json
output_path=${home_dir}/step${step}/annotate/add_data_200_na_100.json
rest_path=${home_dir}/step${step}/rt_aft_anno.json
pr_file=${home_dir}/step${step}/pr_result_max.json
save_na_num=100

cd ..
cd annotate
type=knn_best

python annotate_p.py \
  --data_path ${emb_data_path} \
  --rule_path ${emb_rule_path} \
  --tmp_path ${tmp_path} \
  --output_path ${output_path} \
  --rest_path ${rest_path} \
  --topk ${top_num} \
  --save_na_num ${save_na_num} \
  --mask_num ${mask_num} \
  --rel2id_path ${rel2id_file} \
  --pr_res_path ${pr_file} \
  --type ${type} \
  --threshold ${threshold} \
  --is_sort