#!/bin/bash
#SBATCH --job-name=weirubin_train_mlpmemory_offline_gpt2-large-best-knnxl
#SBATCH --partition=plm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=./logs/%x.log

BASE_MODEL_NAME=large
ALPHA=0.5
DATASET=/mnt/petrelfs/linzhouhan/weirubin/projects/MLPMemory/dataset/wikitext-gpt2
ACCELERATE_CONFIG=./accelerate_config/8_card.yaml
MODEL=/mnt/petrelfs/linzhouhan/weirubin/models/gpt2-large-best-finetuned-wikitext103
KNN_SAVE_PATH=/mnt/petrelfs/linzhouhan/weirubin/projects/MLPMemory/dstore/gpt2-xl/wikitext/knn_gpt2_train_1600.arrow
KNN_DSTORE_PATH=/mnt/petrelfs/linzhouhan/weirubin/projects/MLPMemory/dstore/gpt2-large-best/wikitext/dstore_gpt2_train_1280.arrow
OUTPUT_DIR=./checkpoints/gpt2-large-best-knnxl-offline

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m \
    train_mlpmem_offline \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --dataset_split_name train \
    --knn_save_path ${KNN_SAVE_PATH} \
    --knn_dstore_path ${KNN_DSTORE_PATH} \
    --base_model_name ${BASE_MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate 5e-4 \
    --lr_scheduler_type linear \
    --alpha ${ALPHA} \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_warmup_steps 2000 \
    --num_train_epochs 100 \
    --seed 42 \
    --checkpointing_steps "epoch" \
    --report_to none