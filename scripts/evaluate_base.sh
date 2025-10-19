DATASET=/mnt/petrelfs/linzhouhan/weirubin/projects/MLPMemory/dataset/wikitext-gpt2
MODEL=/mnt/petrelfs/linzhouhan/weirubin/models/gpt2-large
OUTPUT_DIR=tmp/gpt2-large/

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python \
    -m \
    train_base \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --per_device_eval_batch_size 16 \
    --do_eval \
    --eval_subset test \
    --output_dir ${OUTPUT_DIR} \
    --report_to none