#!/bin/bash
#SBATCH --job-name=nanogpt-needle-nope-seq2048-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=results/nanogpt-needle-nope-seq2048-ascend-%j.out
#SBATCH --error=results/nanogpt-needle-nope-seq2048-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# Trains needle-in-haystack at seq=2048 with NoPE for both attn variants.
# NoPE removes the learned positional embedding so the same checkpoint can
# be evaluated at longer seq_len via Triton fwd kernel (eval-long track).
SEQ=2048
ARR=$((SEQ - 8))

for SPEC in "softmax 1.0" "stieltjes 4.0"; do
    set -- $SPEC
    ATTN=$1
    Q=$2
    OUT_DIR="results/needle_${ATTN}_q${Q}_seq${SEQ}_nope_ascend"
    mkdir -p "${OUT_DIR}"
    echo "=== needle ${ATTN} q=${Q} seq=${SEQ} NoPE ==="
    python nanogpt/train.py \
        --task needle --attn "${ATTN}" --q "${Q}" \
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
