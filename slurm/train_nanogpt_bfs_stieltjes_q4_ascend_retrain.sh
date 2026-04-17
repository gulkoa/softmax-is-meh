#!/bin/bash
#SBATCH --job-name=nanogpt-bfs-stj-q4-retrain
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=36G
#SBATCH --time=01:30:00
#SBATCH --output=results/nanogpt-bfs-stj-q4-ascend-retrain-%j.out
#SBATCH --error=results/nanogpt-bfs-stj-q4-ascend-retrain-%j.err

# Corrected-init retrain for tab:accuracy-parity (bfs stj q=4)
# Uses lr=1e-4 per v2 pattern (fixes NaN divergence at lr=3e-4)
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/bfs_stieltjes_q4.0_ascend_retrain"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task bfs --attn stieltjes --q 4.0 --lr 1e-4 \
    --out "${OUT_DIR}" --epochs 50

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task bfs --attn stieltjes --q 4.0 \
    --out "${OUT_DIR}/accuracy_fixed.json"

python nanogpt/analyze.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task bfs --attn stieltjes --q 4.0 \
    --out "${OUT_DIR}/analysis_ascend"
