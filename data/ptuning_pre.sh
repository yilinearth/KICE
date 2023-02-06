step=0
#get ground truth concept from microsoft concept graph
#input
dataset=tacred
train_path=step${step}/RE_model_data/train.json
dev_path=step${step}/RE_model_data/dev.json
test_path=${dataset}/my_test.json

#output
mkdir step${step}/ptuning
mkdir step${step}/ptuning/multi_cls
out_train_path=step${step}/ptuning/multi_cls/train.json
out_dev_path=step${step}/ptuning/multi_cls/dev.json
out_test_path=step${step}/ptuning/multi_cls/test.json
con2id_path=step${step}/ptuning/multi_cls/con2id.json

python get_concept.py \
  --train_path ${train_path} \
  --dev_path ${dev_path} \
  --test_path ${test_path} \
  --out_train_path ${out_train_path} \
  --out_dev_path ${out_dev_path} \
  --out_test_path ${out_test_path} \
  --con2id_path ${con2id_path}

#get ground truth data for no relation classifier
mkdir step${step}/ptuning/bool_cls
python deal_bool_data.py