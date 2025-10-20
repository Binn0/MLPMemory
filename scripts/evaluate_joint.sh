DATASET=/path/to/dataset/wikitext-gpt2
# gpt2-large finetuned on wikitext103 can be downloaded in https://huggingface.co/Rubin-Wei/gpt2-large-finetuned-wikitext103
MODEL=/path/to/gpt2-large-finetuned-wikitext103
# gpt2-large mlp memory can be downloaded in https://huggingface.co/Rubin-Wei/MLPMemory-gpt2-large
KNN_PATH=/path/to/mlp/memory
OUTPUT_DIR=tmp/

python -m \
    evaluate_joint \
    --do_test \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --dataset_split_name test \
    --per_device_eval_batch_size 8 \
    --output_dir ${OUTPUT_DIR} \
    --knn_temp 1 \
    --lmbda 0.25 \
    --knn_generator_path ${KNN_PATH} \
    --report_to none