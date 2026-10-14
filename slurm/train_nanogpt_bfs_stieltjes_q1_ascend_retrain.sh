#!/bin/bash
#SBATCH --job-name=nanogpt-bfs-stj-q1-retrain
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=36G
#SBATCH --time=01:30:00
#SBATCH --output=results/nanogpt-bfs-stj-q1-ascend-retrain-%j.out
#SBATCH --error=results/nanogpt-bfs-stj-q1-ascend-retrain-%j.err

# Post-fix retrain for tab:accuracy-parity (bfs stj q=1).
# Replaces pre-fix 0.816 cell that used the all-positions val_accuracy metric.
# lr=3e-4 matches v3 protocol that produced the original 0.816; the new
# train.py reports output-only val_accuracy, making this comparable to the
# post-fix softmax (0.995), q=4 (0.992), q=8 (0.991) cells.
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/bfs_stieltjes_q1.0_ascend_retrain"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task bfs --attn stieltjes --q 1.0 --lr 3e-4 \
    --out "${OUT_DIR}" --epochs 50

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task bfs --attn stieltjes --q 1.0 \
    --out "${OUT_DIR}/accuracy_fixed.json"

python nanogpt/analyze.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task bfs --attn stieltjes --q 1.0 \
    --out "${OUT_DIR}/analysis_ascend"
