export CUDA_VISIBLE_DEVICES=0

step=$1
old_step=$2
dataset=tacred
topk=3

rel2id_file=../data/${dataset}/my_rel2id.json
candidate_file=../data/step${step}/can_rules.json
rest_file=../data/step${step}/rt_aft_pred.json
ckpt_file=output/step${old_step}/model.ckpt

python pred.py --max_epochs=4  --num_workers=8 \
    --model_name_or_path roberta-base \
    --accumulate_grad_batches 4 \
    --batch_size 16 \
    --data_dir ../data/step${old_step} \
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
    --candidate_file ${candidate_file} \
    --rest_file ${rest_file} \
    --ckpt_file ${ckpt_file} \
    --rel2id_file ${rel2id_file} \
    --dataset ${dataset}
