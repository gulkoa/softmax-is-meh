#!/bin/bash
# Post-URGENT expansion: 6-layer high-q rescue test with constant lambda_0=1.1 init.
# Tests whether the init-fix rescue observed on binary_search (q=8: 0.9996, q=16: 0.9902)
# generalizes to other tasks, or is structural / task-specific.
# Env vars: TASK (needle/max/sorting/bfs), Q (8.0 or 16.0)
#SBATCH --job-name=task-highq-initfix
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/task-highq-initfix-%j.out
#SBATCH --error=results/task-highq-initfix-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/${TASK}_stieltjes_q${Q}_initfix_ascend"
mkdir -p "${OUT_DIR}"
echo "=== 6-job expansion: ${TASK} 6-layer stieltjes q=${Q} post-fix init ==="

python nanogpt/train.py \
    --task "${TASK}" --attn stieltjes --q "${Q}" \
    --epochs 30 \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task "${TASK}" --attn stieltjes --q "${Q}" \
    --out "${OUT_DIR}/accuracy_fixed.json"

echo "DONE ${TASK} q=${Q}"
