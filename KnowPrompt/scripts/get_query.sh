export CUDA_VISIBLE_DEVICES=0
knowl_dir=../data/
topk=20
query_rule_num=40
step=$1
old_step=$2
dataset=tacred

total_path=${knowl_dir}/step${old_step}/rt_aft_pred.json
boot_path=${knowl_dir}/step${old_step}/rule_data/boot_rules.json
ckpt_file=output/step${old_step}/model.ckpt
rel2id_file=../data/${dataset}/my_rel2id.json

rest_file=${knowl_dir}/step${step}/rest_aft_act.json
query_unlabel_path=${knowl_dir}/step${step}/active/query_unlabel_rules.json
query_boot_path=${knowl_dir}/step${step}/active/query_boot_rules.json
conf_path=${knowl_dir}/step${step}/active/confuse_rule.json

python act_query.py --max_epochs=4  --num_workers=8 \
    --model_name_or_path roberta-base \
    --accumulate_grad_batches 4 \
    --batch_size 16 \
    --data_dir ../data/step${old_step} \
    --new_data_dir ../data/step${step} \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80_pred \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 4e-5 \
    --topk ${topk} \
    --query_rule_num ${query_rule_num} \
    --rest_file ${rest_file} \
    --unlabel_file ${total_path} \
    --confuse_file ${conf_path} \
    --query_file ${query_unlabel_path} \
    --query_boot_file ${query_boot_path} \
    --boot_rule_file ${boot_path} \
    --ckpt_file ${ckpt_file} \
    --rel2id_file ${rel2id_file} \
    --dataset ${dataset} \
    --isSelfFix


new_rule_path=../data/step${step}/act_rules.json
type=add_query_rules
python combine.py \
  --old_rule_file ${boot_path} \
  --query_unlabel_file ${query_unlabel_path} \
  --query_boot_file ${query_boot_path} \
  --new_rule_file ${new_rule_path} \
  --type ${type}
