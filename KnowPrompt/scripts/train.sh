export CUDA_VISIBLE_DEVICES=1
step=$1
dataset=tacred
python main.py --max_epochs=4  --num_workers=8 \
    --model_name_or_path roberta-base \
    --accumulate_grad_batches 4 \
    --batch_size 16 \
    --data_dir ../data/step${step} \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 4e-5 \
    --step ${step} \
    --dataset ${dataset}
#max_seq_length=256
#batch_size=16