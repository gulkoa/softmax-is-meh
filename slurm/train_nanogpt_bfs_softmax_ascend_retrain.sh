#!/bin/bash
#SBATCH --job-name=nanogpt-bfs-softmax-ascend-retrain
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=results/nanogpt-bfs-softmax-ascend-retrain-%j.out
#SBATCH --error=results/nanogpt-bfs-softmax-ascend-retrain-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/bfs_softmax_ascend_retrain"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task bfs --attn softmax \
    --out "${OUT_DIR}" --epochs 50

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task bfs --attn softmax --q 1.0 \
    --out "${OUT_DIR}/accuracy_fixed.json"

python nanogpt/analyze.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task bfs --attn softmax \
    --out "${OUT_DIR}/analysis_ascend"
