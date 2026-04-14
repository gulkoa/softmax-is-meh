#!/bin/bash
#SBATCH --job-name=nanogpt-subtle-needle-highq-nope-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=results/nanogpt-subtle-needle-highq-nope-ascend-%j.out
#SBATCH --error=results/nanogpt-subtle-needle-highq-nope-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# High-q sweep on subtle-needle. Expected: all fail to train from scratch
# (per q=8 untrainability finding). Extends phase-transition characterization.
SEQ=2048
ARR=$((SEQ - 8))

for Q in 16.0 32.0 64.0; do
    OUT_DIR="results/subtle_needle_stieltjes_q${Q}_seq${SEQ}_nope_ascend"
    mkdir -p "${OUT_DIR}"
    echo "=== subtle-needle stj q=${Q} seq=${SEQ} NoPE ==="
    python nanogpt/train.py \
        --task needle --needle-margin subtle \
        --attn stieltjes --q "${Q}" \
        --pos-enc none \
        --seq-len "${SEQ}" --max-arr-len "${ARR}" \
        --lr 1e-4 --epochs 30 --batch-size 4 \
        --train-samples 20000 --val-samples 2000 \
        --stieltjes-num-iter 3 \
        --out "${OUT_DIR}"

    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task needle --attn stieltjes --q "${Q}" \
        --out "${OUT_DIR}/accuracy_fixed.json"
done
