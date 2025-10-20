TOKENIZER="/path/to/models/gpt2-large-finetuned-wikitext103"
OUTPUT_DIR=./dataset/wikitext-gpt2

python utils/preprocess_dataset.py \
    --dataset_name /mnt/petrelfs/linzhouhan/weirubin/datasets/wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --tokenizer_path ${TOKENIZER} \
    --output_dir ${OUTPUT_DIR} \
    --num_proc 1