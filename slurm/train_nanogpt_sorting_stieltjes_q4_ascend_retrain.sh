#!/bin/bash
#SBATCH --job-name=nanogpt-sorting-stj-q4-retrain
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=36G
#SBATCH --time=01:30:00
#SBATCH --output=results/nanogpt-sorting-stj-q4-ascend-retrain-%j.out
#SBATCH --error=results/nanogpt-sorting-stj-q4-ascend-retrain-%j.err

# Corrected-init retrain for tab:accuracy-parity (sorting stj q=4)
# Uses lr=1e-4 to be consistent with corrected bfs/bsearch retrains
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/sorting_stieltjes_q4.0_ascend_retrain"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task sorting --attn stieltjes --q 4.0 --lr 1e-4 \
    --out "${OUT_DIR}" --epochs 50

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task sorting --attn stieltjes --q 4.0 \
    --out "${OUT_DIR}/accuracy_fixed.json"

python nanogpt/analyze.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task sorting --attn stieltjes --q 4.0 \
    --out "${OUT_DIR}/analysis_ascend"
