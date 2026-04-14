#!/bin/bash
#SBATCH --job-name=nanogpt-needle-softmax-16k-h100
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=36G
#SBATCH --time=12:00:00
#SBATCH --output=results/nanogpt-needle-softmax-16k-h100-%j.out
#SBATCH --error=results/nanogpt-needle-softmax-16k-h100-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

SEQ=16384
ARR=$((SEQ - 8))
OUT_DIR="results/needle_softmax_seq${SEQ}_cardinal_h100"
mkdir -p "${OUT_DIR}"

echo "=== needle softmax seq_len=${SEQ} cardinal h100 ==="
nvidia-smi

python nanogpt/train.py \
    --task needle --attn softmax \
    --seq-len "${SEQ}" --max-arr-len "${ARR}" \
    --lr 1e-4 --epochs 15 --batch-size 1 \
    --train-samples 4000 --val-samples 500 \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task needle --attn softmax --q 1.0 \
    --seq-len "${SEQ}" --val-samples 500 \
    --out "${OUT_DIR}/accuracy_fixed.json"
