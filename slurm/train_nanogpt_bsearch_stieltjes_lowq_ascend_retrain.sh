#!/bin/bash
#SBATCH --job-name=nanogpt-bsearch-stj-lowq-ascend-retrain
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=results/nanogpt-bsearch-stj-lowq-ascend-retrain-%j.out
#SBATCH --error=results/nanogpt-bsearch-stj-lowq-ascend-retrain-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

for Q in 1.0 2.0 4.0; do
    OUT_DIR="results/binary_search_stieltjes_q${Q}_ascend_retrain"
    mkdir -p "${OUT_DIR}"
    echo "=== Training binary_search stieltjes q=${Q} ==="
    python nanogpt/train.py \
        --task binary_search --attn stieltjes --q "${Q}" \
        --out "${OUT_DIR}" --epochs 50

    echo "=== eval_accuracy q=${Q} ==="
    python nanogpt/eval_accuracy.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task binary_search --attn stieltjes --q "${Q}" \
        --out "${OUT_DIR}/accuracy_fixed.json"

    echo "=== analyze q=${Q} ==="
    python nanogpt/analyze.py \
        --checkpoint "${OUT_DIR}/model.pt" \
        --task binary_search --attn stieltjes --q "${Q}" \
        --out "${OUT_DIR}/analysis_ascend"
done
