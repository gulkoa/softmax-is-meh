#!/bin/bash
#SBATCH --job-name=nanogpt-subtle-needle-nope-seq2048-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=results/nanogpt-subtle-needle-nope-seq2048-ascend-%j.out
#SBATCH --error=results/nanogpt-subtle-needle-nope-seq2048-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# Subtle-needle: needle value = max(distractors) + 1. Margin of exactly 1.
# This is the regime where softmax attention dilution should actually bite at
# long context. NoPE so the trained model can be eval'd at longer seq via
# Triton fwd kernel.
SEQ=2048
ARR=$((SEQ - 8))

for SPEC in "softmax 1.0" "stieltjes 4.0"; do
    set -- $SPEC
    ATTN=$1
    Q=$2
    OUT_DIR="results/subtle_needle_${ATTN}_q${Q}_seq${SEQ}_nope_ascend"
    mkdir -p "${OUT_DIR}"
    echo "=== subtle_needle ${ATTN} q=${Q} seq=${SEQ} NoPE ==="
    python nanogpt/train.py \
        --task needle --needle-margin subtle \
        --attn "${ATTN}" --q "${Q}" \
        --pos-enc none \
        --seq-len "${SEQ}" --max-arr-len "${ARR}" \
        --lr 1e-4 --epochs 30 --batch-size 4 \
        --train-samples 20000 --val-samples 2000 \
        --out "${OUT_DIR}"

    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task needle --attn "${ATTN}" --q "${Q}" \
        --out "${OUT_DIR}/accuracy_fixed.json"
done
