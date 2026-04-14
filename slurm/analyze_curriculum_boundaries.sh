#!/bin/bash
#SBATCH --job-name=analyze-curriculum-boundaries
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/analyze-curriculum-boundaries-%j.out
#SBATCH --error=results/analyze-curriculum-boundaries-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

RUN_DIR="results/binary_search_curriculum_q1to8_ascend"

echo "=== analyze epoch 25 (start of q=4 phase) ==="
python nanogpt/analyze.py \
    --checkpoint "${RUN_DIR}/model_epoch025_q4.pt" \
    --task binary_search --attn stieltjes --q 4.0 \
    --seq-len 128 --max-arr-len 16 --max-val 64 \
    --out "${RUN_DIR}/analysis_ep025_q4"

echo "=== analyze epoch 037 (start of q=8 phase, post-collapse) ==="
python nanogpt/analyze.py \
    --checkpoint "${RUN_DIR}/model_epoch037_q8.pt" \
    --task binary_search --attn stieltjes --q 8.0 \
    --seq-len 128 --max-arr-len 16 --max-val 64 \
    --out "${RUN_DIR}/analysis_ep037_q8"
