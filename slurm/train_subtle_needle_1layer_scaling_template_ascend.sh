#!/bin/bash
# Ascend mirror: 1-layer subtle-needle scaling sweep.
# Env vars: ATTN (stieltjes|softmax), Q, SEQ, BS
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

ARR=$((SEQ - 8))
if [ "$ATTN" = "stieltjes" ]; then
    TAG="stieltjes_q${Q}"
    EXTRA_TRAIN="--attn stieltjes --q ${Q}"
    EXTRA_EVAL="--attn stieltjes --q ${Q}"
else
    TAG="softmax"
    EXTRA_TRAIN="--attn softmax"
    EXTRA_EVAL="--attn softmax --q 1.0"
fi

OUT_DIR="results/subtle_needle_1layer_${TAG}_seq${SEQ}_nope_ascend"
mkdir -p "${OUT_DIR}"

echo "=== subtle-needle 1-layer ${TAG} seq=${SEQ} NoPE ascend ==="
nvidia-smi

python -u nanogpt/train.py \
    --task needle --needle-margin subtle --max-arr-len "${ARR}" \
    ${EXTRA_TRAIN} \
    --pos-enc none \
    --seq-len "${SEQ}" \
    --lr 3e-4 --epochs 30 --batch-size "${BS}" \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head 6 --n-embd 384 \
    --dtype bf16 \
    --out "${OUT_DIR}"

for EVAL_SEQ in ${SEQ} $((SEQ * 2)) $((SEQ * 4)); do
    if [ "${EVAL_SEQ}" -gt 16384 ]; then continue; fi
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task needle --needle-margin subtle ${EXTRA_EVAL} \
        --seq-len "${EVAL_SEQ}" --max-arr-len $((EVAL_SEQ - 8)) --val-samples 2000 \
        --out "${OUT_DIR}/accuracy_eval_seq${EVAL_SEQ}_scaledarr.json" || true
done
