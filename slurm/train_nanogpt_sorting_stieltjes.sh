#!/bin/bash
#SBATCH --job-name=nanogpt-sort-stieltjes
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/nanogpt-sort-stieltjes-%j.out
#SBATCH --error=results/nanogpt-sort-stieltjes-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

for Q in 1.0 2.0 4.0 8.0; do
    OUTDIR="results/sorting_stieltjes_q${Q}"
    mkdir -p "$OUTDIR"
    echo "=== Training sorting with stieltjes q=$Q ==="
    python nanogpt/train.py --task sorting --attn stieltjes --q "$Q" --out "$OUTDIR" --epochs 50
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" \
        --task sorting --attn stieltjes --q "$Q" --out "$OUTDIR/analysis"
