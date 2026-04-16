#!/bin/bash
# P3: multi-seed max retrieval at specified n_embd to firm up crossover story.
# Env vars: ATTN, Q, NEMBD, SEED
# NHEAD=1 for n_embd in {8,12,16}; 4 for larger. Here we only use n_embd<=16.
#SBATCH --job-name=max-embd-multiseed
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
NHEAD=1

if [ "$ATTN" = "stieltjes" ]; then
    TAG="stieltjes_q${Q}"
    EXTRA="--attn stieltjes --q ${Q}"
else
    TAG="softmax"
    EXTRA="--attn softmax"
fi

OUT_DIR="results/max_1layer_${TAG}_seq${SEQ}_embd${NEMBD}_h${NHEAD}_seed${SEED}_initfix_ascend"
mkdir -p "${OUT_DIR}"

echo "=== MAX n_embd=${NEMBD} multiseed: ${TAG} seed=${SEED} ==="

python -u nanogpt/train.py \
    --task max --max-arr-len "${ARR}" \
    ${EXTRA} \
    --pos-enc none \
    --seq-len "${SEQ}" \
    --lr 3e-4 --epochs 30 --batch-size "${BS}" \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --n-layer 1 --n-head ${NHEAD} --n-embd ${NEMBD} \
    --dtype bf16 \
    --seed "${SEED}" \
    --out "${OUT_DIR}"

EXTRA_EVAL="${EXTRA}"
[ "$ATTN" = "softmax" ] && EXTRA_EVAL="--attn softmax --q 1.0"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task max ${EXTRA_EVAL} \
    --seq-len 128 --max-arr-len "${ARR}" --val-samples 1000 \
    --batch-size 8 \
    --n-layer 1 --n-head ${NHEAD} --n-embd ${NEMBD} \
    --first-n-output 1 \
    --out "${OUT_DIR}/accuracy_eval_seq128_arr120_first1.json" || echo "FAIL eval 128"

echo "DONE ${OUT_DIR}"
