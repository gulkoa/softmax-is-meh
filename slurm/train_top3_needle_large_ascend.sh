#!/bin/bash
# Top-3 needle on LARGER inputs: seq=512, max_arr_len=500.
# More distractors + 3 outputs forces stronger multi-modal attention.
# Env vars: ATTN, Q, NHEAD (default 6), NEMBD (default 384)
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

SEQ=512
ARR=500
BS=8

if [ "$ATTN" = "stieltjes" ]; then
    TAG="stieltjes_q${Q}"
    EXTRA="--attn stieltjes --q ${Q}"
else
    TAG="softmax"
    EXTRA="--attn softmax"
fi

OUT_DIR="results/top3_needle_1layer_${TAG}_seq${SEQ}_arr${ARR}_ascend"
mkdir -p "${OUT_DIR}"

echo "=== TOP3 needle: ${TAG} seq=${SEQ} arr=${ARR} ==="

python -u nanogpt/train.py \
    --task top3_needle --needle-margin subtle --max-arr-len "${ARR}" \
    ${EXTRA} \
    --pos-enc none \
    --seq-len "${SEQ}" \
    --lr 3e-4 --epochs 30 --batch-size "${BS}" \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head ${NHEAD:-6} --n-embd ${NEMBD:-384} \
    --dtype bf16 \
    --out "${OUT_DIR}"

# Eval at fixed arr=500 across multiple seqs
EXTRA_EVAL="${EXTRA}"
[ "$ATTN" = "softmax" ] && EXTRA_EVAL="--attn softmax --q 1.0"

for ES in 512 1024 2048 4096 8192; do
    if   [ "$ES" -le 1024 ]; then EBS=4
    elif [ "$ES" -le 4096 ]; then EBS=2
    else                          EBS=1
    fi
    TRITON_FLAG=""
    [ "$ATTN" = "stieltjes" ] && [ "$ES" -ge 8192 ] && TRITON_FLAG="--use-triton"
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task top3_needle --needle-margin subtle ${EXTRA_EVAL} \
        --seq-len "$ES" --max-arr-len "$ARR" --val-samples 1000 \
        --batch-size "$EBS" $TRITON_FLAG \
        --out "${OUT_DIR}/accuracy_eval_seq${ES}_arr${ARR}.json" || echo "FAIL eval ${ES}"
done

echo "DONE ${OUT_DIR}"
