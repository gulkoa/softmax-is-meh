#!/bin/bash
#SBATCH --job-name=nanogpt-max-longctx-q24-nope-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=results/nanogpt-max-longctx-q24-nope-ascend-%j.out
#SBATCH --error=results/nanogpt-max-longctx-q24-nope-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

SEQ=2048
ARR=$((SEQ - 8))
Q=24.0
OUT_DIR="results/max_longctx_stieltjes_q${Q}_seq${SEQ}_nope_ascend"
mkdir -p "${OUT_DIR}"

echo "=== max-longctx stj q=${Q} seq=${SEQ} NoPE ==="
python nanogpt/train.py \
    --task max --max-arr-len "${ARR}" \
    --attn stieltjes --q "${Q}" \
    --pos-enc none \
    --seq-len "${SEQ}" \
    --lr 1e-4 --epochs 30 --batch-size 4 \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task max --attn stieltjes --q "${Q}" \
    --out "${OUT_DIR}/accuracy_fixed.json"
