#!/bin/bash
# 1-HEAD toy model for MAX retrieval (direct Velickovic replication).
# Env vars: ATTN (stieltjes|softmax), Q, TRAIN_SEQ (default 128), N_EMBD (default 64)
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
N_EMBD="${N_EMBD:-64}"
ARR=$((TRAIN_SEQ - 8))

if [ "$ATTN" = "stieltjes" ]; then
    TAG="stieltjes_q${Q}"
    EXTRA_TRAIN="--attn stieltjes --q ${Q}"
    EXTRA_EVAL="--attn stieltjes --q ${Q} --use-triton"
else
    TAG="softmax"
    EXTRA_TRAIN="--attn softmax"
    EXTRA_EVAL="--attn softmax --q 1.0"
fi

OUT_DIR="results/max_1head_${TAG}_seq${TRAIN_SEQ}_d${N_EMBD}_nope_ascend"
mkdir -p "${OUT_DIR}"

echo "=== max 1-head toy ${TAG} seq=${TRAIN_SEQ} d=${N_EMBD} NoPE ==="
nvidia-smi

python -u nanogpt/train.py \
    --task max --max-arr-len "${ARR}" \
    ${EXTRA_TRAIN} \
    --pos-enc none \
    --seq-len "${TRAIN_SEQ}" \
    --lr 3e-4 --epochs 50 --batch-size 32 \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head 1 --n-embd "${N_EMBD}" \
    --dtype bf16 \
    --out "${OUT_DIR}"

# Eval at 1×, 4×, 16×, 64×, 128× (cap 16384)
for EVAL_SEQ in ${TRAIN_SEQ} $((TRAIN_SEQ * 4)) $((TRAIN_SEQ * 16)) $((TRAIN_SEQ * 64)) $((TRAIN_SEQ * 128)); do
    if [ "${EVAL_SEQ}" -gt 16384 ]; then continue; fi
    EVAL_ARR=$((EVAL_SEQ - 8))
    echo "=== eval at seq=${EVAL_SEQ} (${EVAL_SEQ}/${TRAIN_SEQ}x extrap) ==="
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task max ${EXTRA_EVAL} \
        --seq-len "${EVAL_SEQ}" --max-arr-len "${EVAL_ARR}" \
        --val-samples 2000 --batch-size 1 \
        --out "${OUT_DIR}/accuracy_eval_seq${EVAL_SEQ}_scaledarr.json" || true
done
