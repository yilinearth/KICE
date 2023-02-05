home_dir=../data

#input
data_path=$1
dataset=tacred
rel2id_file=${home_dir}/${dataset}/my_rel2id.json
mask_num=3
#output
emb_data_path=$2

#get embedding
python embedding.py \
  --data_path ${data_path} \
  --output_path ${emb_data_path} \
  --mask_num ${mask_num}

