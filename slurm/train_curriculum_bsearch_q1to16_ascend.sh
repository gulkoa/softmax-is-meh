#!/bin/bash
#SBATCH --job-name=nanogpt-curriculum-bsearch-q1to16-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=results/nanogpt-curriculum-bsearch-q1to16-ascend-%j.out
#SBATCH --error=results/nanogpt-curriculum-bsearch-q1to16-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

OUT_DIR="results/binary_search_curriculum_q1to16_ascend"
mkdir -p "${OUT_DIR}"

# Schedule: 10 epochs each at q=1,2,4,8,16 → 50 total
python nanogpt/train_curriculum.py \
    --task binary_search --attn stieltjes \
    --q-schedule "1@1,2@11,4@21,8@31,16@41" \
    --out "${OUT_DIR}" --epochs 50 --eval-each-block

# Final accuracy (model ends at q=16)
python nanogpt/eval_accuracy.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task binary_search --attn stieltjes --q 16.0 \
    --out "${OUT_DIR}/accuracy_fixed.json"

python nanogpt/analyze.py \
    --checkpoint "${OUT_DIR}/model.pt" \
    --task binary_search --attn stieltjes --q 16.0 \
    --out "${OUT_DIR}/analysis_ascend"
