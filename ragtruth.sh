#!/bin/bash
# run_icr_ragtruth_all.sh
# ICR Probe on RAGTruth: 2 detection models × 3 task types = 6 runs
# Each run: sentence-level AUROC/F1 + token-level detection (20 samples)
#
# Models:
#   1. /gz-fs/models/Meta-Llama-3.1-8B-Instruct
#   2. /gz-fs/models/Qwen/Qwen2.5-7B
#
# Data: RAGTruth responses by llama-2-7b-chat
#
# Usage: bash run_icr_ragtruth_all.sh

set -e

# ============================================
# Configuration
# ============================================
RAGTRUTH_DIR="../../data/RAGTruth/dataset"
DATA_MODEL="llama-2-7b-chat"
MAX_LENGTH=1024
GPU_ID=0

# ICR params (paper defaults)
ICR_TOP_P=0.1
ICR_POOLING="mean"
PROBE_EPOCHS=100
PROBE_BATCH=32
PROBE_LR=5e-4
TOKEN_SAMPLES=20

TASK_TYPES=("QA" "Summary" "Data2txt")
BASE_OUTPUT="./profiling_results"

# Model list: short_name|full_path
MODEL_LIST=(
    "llama3.1-8B|/gz-fs/models/Meta-Llama-3.1-8B-Instruct"
    "qwen2.5-7B|/gz-fs/models/Qwen/Qwen2.5-7B"
)

# ============================================
# Run function
# ============================================
run_one() {
    local MODEL_KEY=$1
    local MODEL_PATH=$2
    local TASK=$3
    local DIR="${BASE_OUTPUT}/icr_${MODEL_KEY}_${DATA_MODEL}_${TASK}"

    echo ""
    echo "============================================"
    echo "  Detection model:  ${MODEL_KEY}"
    echo "  Model path:       ${MODEL_PATH}"
    echo "  Data model:       ${DATA_MODEL}"
    echo "  Task:             ${TASK}"
    echo "  Token-level:      ${TOKEN_SAMPLES} samples"
    echo "  Output:           ${DIR}"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python run_profiled.py \
        --model_name "${MODEL_PATH}" \
        --ragtruth_data_dir "${RAGTRUTH_DIR}" \
        --ragtruth_model_filter ${DATA_MODEL} \
        --ragtruth_task_types ${TASK} \
        --ragtruth_max_length ${MAX_LENGTH} \
        --icr_top_p ${ICR_TOP_P} \
        --icr_pooling ${ICR_POOLING} \
        --probe_num_epochs ${PROBE_EPOCHS} \
        --probe_batch_size ${PROBE_BATCH} \
        --probe_lr ${PROBE_LR} \
        --token_level_n_samples ${TOKEN_SAMPLES} \
        --profile_output_dir "${DIR}/"

    echo "  Done: ${MODEL_KEY} × ${TASK}"
}

# ============================================
# Main: 2 models × 3 tasks = 6 runs
# ============================================
TOTAL_RUNS=$(( ${#MODEL_LIST[@]} * ${#TASK_TYPES[@]} ))

echo "================================================================"
echo "  ICR Probe × RAGTruth"
echo "  Models:  ${#MODEL_LIST[@]}"
echo "  Tasks:   ${TASK_TYPES[*]}"
echo "  Runs:    ${TOTAL_RUNS}"
echo "  GPU:     ${GPU_ID}"
echo "================================================================"

START_TIME=$(date +%s)
RUN=0

for ENTRY in "${MODEL_LIST[@]}"; do
    IFS='|' read -r MODEL_KEY MODEL_PATH <<< "${ENTRY}"
    for TASK in "${TASK_TYPES[@]}"; do
        RUN=$((RUN + 1))
        echo ""
        echo ">>> [${RUN}/${TOTAL_RUNS}] ${MODEL_KEY} × ${TASK}"
        run_one "${MODEL_KEY}" "${MODEL_PATH}" "${TASK}"
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ============================================
# Summary Table
# ============================================
echo ""
echo "================================================================"
echo "  All ${TOTAL_RUNS} runs complete!  Total: ${ELAPSED}s"
echo "================================================================"
echo ""
printf "  %-15s %-10s %8s %8s\n" "Model" "Task" "AUROC" "Time"
printf "  %-15s %-10s %8s %8s\n" "-----" "----" "-----" "----"

for ENTRY in "${MODEL_LIST[@]}"; do
    IFS='|' read -r MODEL_KEY MODEL_PATH <<< "${ENTRY}"
    for TASK in "${TASK_TYPES[@]}"; do
        DIR="${BASE_OUTPUT}/icr_${MODEL_KEY}_${DATA_MODEL}_${TASK}"
        if [ -f "${DIR}/profiling_summary.json" ]; then
            AUROC=$(python3 -c "
import json; d=json.load(open('${DIR}/profiling_summary.json'))
best=-1
for p in d.get('phases',{}).values():
    a=p.get('best_auroc',-1)
    if a>best: best=a
print(f'{best:.4f}' if best>0 else 'N/A')
" 2>/dev/null || echo "N/A")
            TIME=$(python3 -c "
import json; d=json.load(open('${DIR}/profiling_summary.json'))
print(f\"{d.get('total_time_seconds',0):.0f}s\")
" 2>/dev/null || echo "N/A")
            printf "  %-15s %-10s %8s %8s\n" "${MODEL_KEY}" "${TASK}" "${AUROC}" "${TIME}"
        else
            printf "  %-15s %-10s %8s %8s\n" "${MODEL_KEY}" "${TASK}" "FAILED" "-"
        fi
    done
done

echo ""
echo "Reports:"
for ENTRY in "${MODEL_LIST[@]}"; do
    IFS='|' read -r MODEL_KEY MODEL_PATH <<< "${ENTRY}"
    for TASK in "${TASK_TYPES[@]}"; do
        echo "  cat ${BASE_OUTPUT}/icr_${MODEL_KEY}_${DATA_MODEL}_${TASK}/profiling_report.txt"
    done
done

echo ""
echo "Token-level:"
for ENTRY in "${MODEL_LIST[@]}"; do
    IFS='|' read -r MODEL_KEY MODEL_PATH <<< "${ENTRY}"
    for TASK in "${TASK_TYPES[@]}"; do
        echo "  cat ${BASE_OUTPUT}/icr_${MODEL_KEY}_${DATA_MODEL}_${TASK}/token_level_results.json"
    done
done