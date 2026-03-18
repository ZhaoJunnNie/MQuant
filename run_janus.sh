#!/usr/bin/env bash
set -euo pipefail

tasks=(MME MMBench_DEV_EN SEEDBench_IMG MMMU_DEV_VAL RealWorldQA TextVQA_VAL)
# 固定为用户指定的命令与参数，所有 task 以 --dataset_name 指定

for task in "${tasks[@]}"; do
    echo "Starting run with w_bits=4, a_bits=8, task=$task..."

    python exam/quant_janus.py \
        --model_name Janus-Pro-7B \
        --quant \
        --rotate \
        --rotate_llm \
        --online_llm_hadamard \
        --quant_llm \
        --llm_w_bits 4 \
        --llm_a_bits 8 \
        --llm_static --nsamples 128 --calib_num 128 \
        --dataset_name "$task" \
        2>&1 | tee "logs/quant_janus-\$(date +%Y%m%d_%H%M%S).log"

    echo "Completed run with w_bits=4, a_bits=8, task=$task"
    echo "--------------------------------------------------"
done