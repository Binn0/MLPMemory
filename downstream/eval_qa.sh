#!/bin/bash

MODEL=/path/to/models/Mistral-7B-v0.3
MODEL_NAME=mistral
# MLP Memory can be downloaded in https://huggingface.co/Rubin-Wei/MLPMemory-Mistral-wikipedia
KNN_GENERATOR_PATH=/path/to/mlp/memory

TASKS=("nq" "hotpotqa")
# or hotpotqa

LAMBDA_STRING="0.50"

IFS='+' read -ra LAMBDAS <<< "$LAMBDA_STRING"

if [ ${#LAMBDAS[@]} -gt 8 ]; then
    echo "Error: Too many lambda values (${#LAMBDAS[@]}). Maximum 8 (GPUs 0-7)"
    exit 1
fi

echo "======================================="
echo "Starting parallel evaluation"
echo "Model: $MODEL_NAME"
echo "Lambda values: ${LAMBDAS[@]}"
echo "======================================="

# Loop over all tasks
for TASK in "${TASKS[@]}"; do
    echo ""
    echo "======================================="
    echo "Evaluating task: $TASK"
    echo "======================================="

    if [[ $TASK == "nq" ]]; then
        DATASET_NAME=/mnt/petrelfs/linzhouhan/weirubin/datasets/nq_open
    elif [[ $TASK == "hotpotqa" ]]; then
        DATASET_NAME=/mnt/petrelfs/linzhouhan/weirubin/datasets/hotpot_qa
    else
        echo "Error: Unknown task $TASK"
        exit 1
    fi

    # Function to run evaluation for a specific lambda on a specific GPU
    run_evaluation() {
        local LMBDA=$1
        local GPU_ID=$2

        echo "[GPU $GPU_ID] Starting evaluation with lambda=$LMBDA for task $TASK"

        export CUDA_VISIBLE_DEVICES=$GPU_ID
        LOG_FILE="logs/${TASK}_${MODEL_NAME}_lambda_${LMBDA}_gpu_${GPU_ID}.log"
        mkdir -p logs

        if [[ $TASK == "nq" ]]; then
            python -m downstream.eval_qa \
              --model_name_or_path ${MODEL} \
              --dataset_name ${DATASET_NAME} \
              --ignore_pad_token_for_loss \
              --per_device_eval_batch_size=1 \
              --output_dir downstream/results/nq/neural/${MODEL_NAME}-${LMBDA} \
              --use_neural_knn \
              --knn_generator_path ${KNN_GENERATOR_PATH} \
              --lmbda ${LMBDA} \
              --do_eval \
              --eval_subset validation
        elif [[ $TASK == "hotpotqa" ]]; then
            python -m downstream.eval_qa \
              --model_name_or_path ${MODEL} \
              --dataset_name ${DATASET_NAME} --dataset_config_name distractor \
              --ignore_pad_token_for_loss \
              --per_device_eval_batch_size=1 \
              --output_dir downstream/results/hotpot/neural/${MODEL_NAME}-${LMBDA} \
              --use_neural_knn \
              --knn_generator_path ${KNN_GENERATOR_PATH} \
              --lmbda ${LMBDA} \
              --do_eval \
              --eval_subset validation
        fi

        echo "[GPU $GPU_ID] Completed evaluation with lambda=$LMBDA (log: ${LOG_FILE})"
    }

    export -f run_evaluation
    export MODEL MODEL_NAME KNN_GENERATOR_PATH TASK DATASET_NAME

    # Start parallel evaluations for this task
    PIDS=()
    for i in "${!LAMBDAS[@]}"; do
        LMBDA=${LAMBDAS[$i]}
        GPU_ID=$i
        run_evaluation $LMBDA $GPU_ID &
        PIDS+=($!)
        echo "Started process PID ${PIDS[-1]} for lambda=$LMBDA on GPU $GPU_ID"
    done

    # Wait for all background processes to complete
    echo "Waiting for all evaluations for task $TASK to complete..."
    for pid in "${PIDS[@]}"; do
        wait $pid
    done

    echo "======================================="
    echo "Completed all evaluations for task: $TASK"
    echo "======================================="
done

echo "======================================="
echo "All evaluations for all tasks completed!"
echo "======================================="
