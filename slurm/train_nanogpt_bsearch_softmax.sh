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
_search="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$_search" != "/" ]; do
    if [ -f "$_search/triton/pyproject.toml" ]; then REPO_DIR="$_search"; break; fi
    _search="$(dirname "$_search")"
done
if [ -z "${REPO_DIR:-}" ]; then echo "ERROR: repo root not found" >&2; exit 1; fi

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"
mkdir -p results/binary_search_softmax

python nanogpt/train.py --task binary_search --attn softmax --out results/binary_search_softmax --epochs 50
python nanogpt/analyze.py --checkpoint results/binary_search_softmax/model.pt \
    --task binary_search --attn softmax --out results/binary_search_softmax/analysis
