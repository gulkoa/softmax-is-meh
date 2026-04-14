#!/bin/bash
#SBATCH --job-name=nanogpt-max-longctx-nope-seq2048-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=results/nanogpt-max-longctx-nope-seq2048-ascend-%j.out
#SBATCH --error=results/nanogpt-max-longctx-nope-seq2048-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# Max-finding at long context. Like subtle-needle, this is a +1-margin
# task (expected gap between top two values under uniform sampling) so
# softmax dilution should bite at long context. NoPE so we can eval-long
# via Triton fwd kernel.
SEQ=2048
ARR=$((SEQ - 8))

for SPEC in "softmax 1.0" "stieltjes 4.0"; do
    set -- $SPEC
    ATTN=$1
    Q=$2
    OUT_DIR="results/max_longctx_${ATTN}_q${Q}_seq${SEQ}_nope_ascend"
    mkdir -p "${OUT_DIR}"
    echo "=== max-longctx ${ATTN} q=${Q} seq=${SEQ} NoPE ==="
    python nanogpt/train.py \
        --task max --max-arr-len "${ARR}" \
        --attn "${ATTN}" --q "${Q}" \
        --pos-enc none \
        --seq-len "${SEQ}" \
        --lr 1e-4 --epochs 30 --batch-size 4 \
        --train-samples 20000 --val-samples 2000 \
        --out "${OUT_DIR}"

    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task max --attn "${ATTN}" --q "${Q}" \
        --out "${OUT_DIR}/accuracy_fixed.json"
done
