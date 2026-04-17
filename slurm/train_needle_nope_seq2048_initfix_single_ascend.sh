#!/bin/bash
# P1a REDO: single-Q version because the 2-Q sequential version timed out at 4h
# wall with Q=2 only reaching epoch 7 (0.007 acc — still training).
# Env var: Q (2.0 or 4.0)
#SBATCH --job-name=needle-nope-initfix
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=results/needle-nope-initfix-%j.out
#SBATCH --error=results/needle-nope-initfix-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

SEQ=2048
ARR=$((SEQ - 8))
OUT_DIR="results/needle_stieltjes_q${Q}_seq${SEQ}_nope_initfix_ascend"
mkdir -p "${OUT_DIR}"
echo "=== P1a REDO: needle stieltjes q=${Q} seq=${SEQ} NoPE post-fix init ==="

python nanogpt/train.py \
    --task needle --attn stieltjes --q "${Q}" \
    --pos-enc none \
    --seq-len "${SEQ}" --max-arr-len "${ARR}" \
    --lr 1e-4 --epochs 30 --batch-size 4 \
    --train-samples 20000 --val-samples 2000 \
    --out "${OUT_DIR}"

for ES in 2048 4096 8192 16384; do
    if   [ "$ES" -le 4096 ]; then EBS=2
    else                          EBS=1
    fi
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task needle --attn stieltjes --q "${Q}" \
        --seq-len "$ES" --max-arr-len "$((ES - 8))" --val-samples 1000 \
        --batch-size "$EBS" \
        --out "${OUT_DIR}/accuracy_fixed_eval${ES}.json" || echo "FAIL eval ${ES}"
done

echo "DONE P1a q=${Q}"
