#!/bin/bash
#SBATCH --job-name=nanogpt-bsearch-softmax
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=results/nanogpt-bsearch-softmax-%j.out
#SBATCH --error=results/nanogpt-bsearch-softmax-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"
mkdir -p results/binary_search_softmax

python nanogpt/train.py --task binary_search --attn softmax --out results/binary_search_softmax --epochs 50
python nanogpt/analyze.py --checkpoint results/binary_search_softmax/model.pt \
    --task binary_search --attn softmax --out results/binary_search_softmax/analysis
