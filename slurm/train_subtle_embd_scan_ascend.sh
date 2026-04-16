#!/bin/bash
# Subtle-needle: n_embd scan to find softmax-vs-stj crossover point.
# Env vars: ATTN, Q, NEMBD, NHEAD
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

OUT_DIR="results/subtle_1layer_${TAG}_seq${SEQ}_embd${NEMBD}_h${NHEAD}_initfix_ascend"
mkdir -p "${OUT_DIR}"

echo "=== SUBTLE embd-scan: ${TAG} n_embd=${NEMBD} n_head=${NHEAD} ==="

python -u nanogpt/train.py \
    --task needle --needle-margin subtle --max-arr-len "${ARR}" \
    ${EXTRA} \
    --pos-enc none \
    --seq-len "${SEQ}" \
    --lr 3e-4 --epochs 30 --batch-size "${BS}" \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head ${NHEAD} --n-embd ${NEMBD} \
    --dtype bf16 \
    --out "${OUT_DIR}"

EXTRA_EVAL="${EXTRA}"
[ "$ATTN" = "softmax" ] && EXTRA_EVAL="--attn softmax --q 1.0"

for ES in 128 256 512 1024 2048 4096 8192; do
    if   [ "$ES" -le 1024 ]; then EBS=8
    elif [ "$ES" -le 4096 ]; then EBS=2
    else                          EBS=1
    fi
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task needle --needle-margin subtle ${EXTRA_EVAL} \
        --seq-len "$ES" --max-arr-len "$ARR" --val-samples 1000 \
        --batch-size "$EBS" \
        --n-layer 1 --n-head ${NHEAD} --n-embd ${NEMBD} \
        --out "${OUT_DIR}/accuracy_eval_seq${ES}_arr120.json" || echo "FAIL eval ${ES}"
done

echo "DONE ${OUT_DIR}"
