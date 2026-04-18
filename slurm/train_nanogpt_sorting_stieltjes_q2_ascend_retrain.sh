#!/bin/bash
#SBATCH --job-name=nanogpt-sorting-stj-q2-retrain
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=36G
#SBATCH --time=01:30:00
#SBATCH --output=results/nanogpt-sorting-stj-q2-ascend-retrain-%j.out
#SBATCH --error=results/nanogpt-sorting-stj-q2-ascend-retrain-%j.err

# Post-fix retrain for tab:accuracy-parity (sorting stj q=2).
# Replaces pre-fix 0.551 cell that used the all-positions val_accuracy metric.
# lr=3e-4 matches v3 protocol that produced the original 0.551; the new
# train.py reports output-only val_accuracy, making this comparable to the
# post-fix softmax (0.998), q=4 (0.998), q=8 (0.967) cells.
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/sorting_stieltjes_q2.0_ascend_retrain"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task sorting --attn stieltjes --q 2.0 --lr 3e-4 \
    --out "${OUT_DIR}" --epochs 50

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task sorting --attn stieltjes --q 2.0 \
    --out "${OUT_DIR}/accuracy_fixed.json"

python nanogpt/analyze.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task sorting --attn stieltjes --q 2.0 \
    --out "${OUT_DIR}/analysis_ascend"
