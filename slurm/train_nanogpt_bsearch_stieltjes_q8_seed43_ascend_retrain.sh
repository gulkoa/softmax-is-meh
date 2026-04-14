#!/bin/bash
#SBATCH --job-name=nanogpt-bsearch-stj-q8-seed43-ascend-retrain
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=results/nanogpt-bsearch-stj-q8-seed43-ascend-retrain-%j.out
#SBATCH --error=results/nanogpt-bsearch-stj-q8-seed43-ascend-retrain-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

Q=8.0
SEED=43
OUT_DIR="results/binary_search_stieltjes_q${Q}_seed${SEED}_ascend_retrain"
mkdir -p "${OUT_DIR}"

python nanogpt/train.py \
    --task binary_search --attn stieltjes --q "${Q}" --seed "${SEED}" \
    --out "${OUT_DIR}" --epochs 50

python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task binary_search --attn stieltjes --q "${Q}" --seed "${SEED}" \
    --out "${OUT_DIR}/accuracy_fixed.json"

python nanogpt/analyze.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task binary_search --attn stieltjes --q "${Q}" \
    --out "${OUT_DIR}/analysis_ascend"
