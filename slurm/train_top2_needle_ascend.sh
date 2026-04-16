#!/bin/bash
# Top-2 needle: must output BOTH max and second-max. Multi-modal attention test.
# Hypothesis: softmax exponential collapses to single peak (predicts top-1 well, top-2 poorly).
# Stj polynomial decay preserves multiple modes (predicts both).
# Env vars: ATTN, Q
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

SEQ=128
ARR=120
BS=32

if [ "$ATTN" = "stieltjes" ]; then
    TAG="stieltjes_q${Q}"
    EXTRA="--attn stieltjes --q ${Q}"
else
    TAG="softmax"
    EXTRA="--attn softmax"
fi

SUFFIX="${SUFFIX:-_ascend}"
OUT_DIR="results/top2_needle_1layer_${TAG}_seq${SEQ}${SUFFIX}"
mkdir -p "${OUT_DIR}"

echo "=== TOP2 needle: ${TAG} seq=${SEQ} ==="

python -u nanogpt/train.py \
    --task top2_needle --needle-margin subtle --max-arr-len "${ARR}" \
    ${EXTRA} \
    --pos-enc none \
    --seq-len "${SEQ}" \
    --lr 3e-4 --epochs 30 --batch-size "${BS}" \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head ${NHEAD:-6} --n-embd ${NEMBD:-384} \
    --dtype bf16 \
    --out "${OUT_DIR}"

# Eval at same arr=120 across multiple seqs
EXTRA_EVAL="${EXTRA}"
[ "$ATTN" = "softmax" ] && EXTRA_EVAL="--attn softmax --q 1.0"

for ES in 128 256 512 1024 2048 4096 8192; do
    if   [ "$ES" -le 1024 ]; then EBS=8
    elif [ "$ES" -le 4096 ]; then EBS=2
    else                          EBS=1
    fi
    TRITON_FLAG=""
    [ "$ATTN" = "stieltjes" ] && [ "$ES" -ge 2048 ] && TRITON_FLAG="--use-triton"
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task top2_needle --needle-margin subtle ${EXTRA_EVAL} \
        --seq-len "$ES" --max-arr-len "$ARR" --val-samples 1000 \
        --batch-size "$EBS" $TRITON_FLAG \
        --out "${OUT_DIR}/accuracy_eval_seq${ES}_arr120.json" || echo "FAIL eval ${ES}"
done

echo "DONE ${OUT_DIR}"
