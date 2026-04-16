#!/bin/bash
#SBATCH --job-name=nanogpt-subtle-needle-q24-nope-h100
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=36G
#SBATCH --time=5:00:00
#SBATCH --output=results/nanogpt-subtle-needle-q24-nope-h100-%j.out
#SBATCH --error=results/nanogpt-subtle-needle-q24-nope-h100-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

SEQ=2048
ARR=$((SEQ - 8))
Q=24.0
OUT_DIR="results/subtle_needle_stieltjes_q${Q}_seq${SEQ}_nope_h100"
mkdir -p "${OUT_DIR}"

echo "=== subtle-needle stj q=${Q} seq=${SEQ} NoPE h100 ==="
nvidia-smi

python -u nanogpt/train.py \
    --task needle --needle-margin subtle --max-arr-len "${ARR}" \
    --attn stieltjes --q "${Q}" \
    --pos-enc none \
    --seq-len "${SEQ}" \
    --lr 1e-4 --epochs 30 --batch-size 4 \
    --train-samples 20000 --val-samples 2000 \
    --stieltjes-num-iter 3 \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task needle --needle-margin subtle \
    --attn stieltjes --q "${Q}" \
    --out "${OUT_DIR}/accuracy_fixed.json"
