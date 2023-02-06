export CUDA_VISIBLE_DEVICES=1,2
home_dir=../../data/
step=0
#input
my_data_dir=${home_dir}/step${step}/ptuning/bool_cls
#output: ptuned model
mkdir output/boolre
output_dir=output/boolre/step${step}

python3 cli.py \
  --data_dir ${my_data_dir} \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --task_name boolre \
  --output_dir ${output_dir} \
  --do_eval \
  --do_train \
  --pet_per_gpu_eval_batch_size 8 \
  --pet_per_gpu_train_batch_size 2 \
  --pet_gradient_accumulation_steps 1 \
  --pet_max_seq_length 256 \
  --pet_max_steps 250 \
  --pattern_ids 1 \
  --embed_size 768 \
  --eval_set test \
  --learning_rate 1e-4
