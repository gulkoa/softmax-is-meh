#!/bin/bash
# Train subtle-needle 1-layer at small seq (128), then eval at {128, 512, 2048, 8192}
# to replicate Velickovic et al. 2024 "softmax is not enough" regime.
# Env vars: ATTN (stieltjes|softmax), Q, TRAIN_SEQ (default 128)
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

TRAIN_SEQ="${TRAIN_SEQ:-128}"
ARR=$((TRAIN_SEQ - 8))

if [ "$ATTN" = "stieltjes" ]; then
    TAG="stieltjes_q${Q}"
    EXTRA_TRAIN="--attn stieltjes --q ${Q}"
    EXTRA_EVAL="--attn stieltjes --q ${Q}"
else
    TAG="softmax"
    EXTRA_TRAIN="--attn softmax"
    EXTRA_EVAL="--attn softmax --q 1.0"
fi

OUT_DIR="results/subtle_needle_1layer_${TAG}_seq${TRAIN_SEQ}_nope_ascend"
mkdir -p "${OUT_DIR}"

echo "=== subtle-needle 1-layer ${TAG} train-seq=${TRAIN_SEQ} NoPE ==="
nvidia-smi

python -u nanogpt/train.py \
    --task needle --needle-margin subtle --max-arr-len "${ARR}" \
    ${EXTRA_TRAIN} \
    --pos-enc none \
    --seq-len "${TRAIN_SEQ}" \
    --lr 3e-4 --epochs 30 --batch-size 32 \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head 6 --n-embd 384 \
    --dtype bf16 \
    --out "${OUT_DIR}"

# Eval at extreme extrapolation ratios: 1x, 4x, 16x, 64x
for EVAL_SEQ in ${TRAIN_SEQ} $((TRAIN_SEQ * 4)) $((TRAIN_SEQ * 16)) $((TRAIN_SEQ * 64)); do
    if [ "${EVAL_SEQ}" -gt 8192 ]; then continue; fi
    EVAL_ARR=$((EVAL_SEQ - 8))
    echo "=== eval at seq=${EVAL_SEQ} (${EVAL_SEQ}/${TRAIN_SEQ}x extrap) ==="
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task needle --needle-margin subtle ${EXTRA_EVAL} \
        --seq-len "${EVAL_SEQ}" --max-arr-len "${EVAL_ARR}" --val-samples 2000 \
        --out "${OUT_DIR}/accuracy_eval_seq${EVAL_SEQ}_scaledarr.json" || true
done
