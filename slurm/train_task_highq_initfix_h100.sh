#!/bin/bash
# H100 companion to train_task_highq_initfix_ascend.sh — adds seed=1 (or
# user-specified SEED) runs for multi-seed coverage on the narrative gate.
# Env vars: TASK (needle/max/sorting/bfs/binary_search), Q (e.g. 8.0 or 16.0),
# SEED (default 1).
#SBATCH --job-name=task-highq-initfix-h100
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/task-highq-initfix-h100-%j.out
#SBATCH --error=results/task-highq-initfix-h100-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

: "${TASK:?TASK env var required}"
: "${Q:?Q env var required}"
SEED="${SEED:-1}"

OUT_DIR="results/${TASK}_stieltjes_q${Q}_seed${SEED}_initfix_h100"
mkdir -p "${OUT_DIR}"

echo "=== ${TASK} stieltjes q=${Q} seed=${SEED} post-fix on H100 ==="
nvidia-smi
date

python nanogpt/train.py \
    --task "${TASK}" --attn stieltjes --q "${Q}" \
    --epochs 30 --seed "${SEED}" \
    --out "${OUT_DIR}"

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task "${TASK}" --attn stieltjes --q "${Q}" \
    --out "${OUT_DIR}/accuracy_fixed.json"

date
echo "DONE ${TASK} q=${Q} seed=${SEED} on H100"
