#!/bin/bash
#SBATCH --job-name=nanogpt-bfs-stj-q1-seed1-h100
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=36G
#SBATCH --time=01:30:00
#SBATCH --output=results/nanogpt-bfs-stj-q1-seed1-h100-%j.out
#SBATCH --error=results/nanogpt-bfs-stj-q1-seed1-h100-%j.err

# Seed=1 H100 complement to bfs_stieltjes_q1.0_ascend_retrain (seed=42).
# Matches seed=42 protocol: lr=3e-4, 50 epochs. Completes 2-seed coverage
# for the post-fix bfs stj q=1 Table 1 cell.
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/bfs_stieltjes_q1.0_seed1_h100"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task bfs --attn stieltjes --q 1.0 --lr 3e-4 --seed 1 \
    --out "${OUT_DIR}" --epochs 50

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task bfs --attn stieltjes --q 1.0 \
    --out "${OUT_DIR}/accuracy_fixed.json"
