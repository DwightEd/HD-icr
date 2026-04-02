#!/bin/bash
# ragtruth.sh — ICR Probe on RAGTruth, single 4090-48GB
set -e
RAGTRUTH_DIR="/gz-fs/data/RAGTruth/dataset"
DATA_MODEL="llama-2-7b-chat"
GPU_ID=0
ICR_TOP_K=20
ICR_POOLING="mean"
PROBE_EPOCHS=50
PROBE_BATCH=32
PROBE_LR=5e-4
TOKEN_SAMPLES=20
TASK_TYPES=("QA" "Summary" "Data2txt")
BASE_OUTPUT="./profiling_results"
MODEL_KEY="qwen3-8B"
MODEL_PATH="/gz-fs/models/Qwen/Qwen3-8B"

run_one() {
    local TASK=$1
    local DIR="${BASE_OUTPUT}/icr_${MODEL_KEY}_${DATA_MODEL}_${TASK}"
    echo ""
    echo "============================================"
    echo "  ${MODEL_KEY} × ${TASK}"
    echo "  Output: ${DIR}"
    echo "============================================"
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python run_profiled.py \
        --model_name "${MODEL_PATH}" \
        --ragtruth_data_dir "${RAGTRUTH_DIR}" \
        --ragtruth_model_filter ${DATA_MODEL} \
        --ragtruth_task_types ${TASK} \
        --icr_top_k ${ICR_TOP_K} \
        --icr_pooling ${ICR_POOLING} \
        --probe_num_epochs ${PROBE_EPOCHS} \
        --probe_batch_size ${PROBE_BATCH} \
        --probe_lr ${PROBE_LR} \
        --token_level_n_samples ${TOKEN_SAMPLES} \
        --profile_output_dir "${DIR}/"
    echo "  ✓ ${MODEL_KEY} × ${TASK}"
}

TOTAL=${#TASK_TYPES[@]}
echo "================================================================"
echo "  ICR Probe × RAGTruth — qwen3-8B only"
echo "  Runs: ${TOTAL}"
echo "  top_k=${ICR_TOP_K} epochs=${PROBE_EPOCHS}"
echo "================================================================"
START=$(date +%s)
RUN=0
for T in "${TASK_TYPES[@]}"; do
    RUN=$((RUN+1))
    echo ">>> [${RUN}/${TOTAL}] ${MODEL_KEY} × ${T}"
    run_one "$T"
done
END=$(date +%s)
echo ""
echo "================================================================"
echo "  Done! ${RUN} runs in $((END-START))s"
echo "================================================================"
printf "  %-15s %-10s %8s %6s %8s\n" "Model" "Task" "AUROC" "F1" "Time"
printf "  %-15s %-10s %8s %6s %8s\n" "-----" "----" "-----" "--" "----"
for T in "${TASK_TYPES[@]}"; do
    D="${BASE_OUTPUT}/icr_${MODEL_KEY}_${DATA_MODEL}_${T}"
    if [ -f "${D}/profiling_summary.json" ]; then
        V=$(python3 -c "
import json; d=json.load(open('${D}/profiling_summary.json'))
auroc=-1; f1=-1
for p in d.get('phases',{}).values():
    a=p.get('best_auroc',-1); b=p.get('best_f1',-1)
    if a>auroc: auroc=a
    if b>f1: f1=b
t=d.get('total_time_seconds',0)
print(f'{auroc:.4f}' if auroc>0 else 'N/A', f'{f1:.4f}' if f1>0 else 'N/A', f'{t:.0f}s')
" 2>/dev/null || echo "N/A N/A N/A")
        read AU F1 TM <<< "$V"
        printf "  %-15s %-10s %8s %6s %8s\n" "$MODEL_KEY" "$T" "$AU" "$F1" "$TM"
    else
        printf "  %-15s %-10s %8s %6s %8s\n" "$MODEL_KEY" "$T" "—" "—" "—"
    fi
done