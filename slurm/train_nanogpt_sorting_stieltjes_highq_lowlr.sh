#!/bin/bash
#SBATCH --job-name=nanogpt-sort-hq-lowlr
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/nanogpt-sort-hq-lowlr-%j.out
#SBATCH --error=results/nanogpt-sort-hq-lowlr-%j.err

# High-q sorting with lower learning rate (1e-4 vs 3e-4 default)
# Main highq run had q=16 stuck at random loss — testing if LR was the issue
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

for Q in 16.0 32.0 64.0; do
    OUTDIR="results/sorting_stieltjes_q${Q}_lowlr"
    mkdir -p "$OUTDIR"
    echo "=== Training sorting with stieltjes q=$Q (lr=1e-4) ==="
    python nanogpt/train.py --task sorting --attn stieltjes --q "$Q" --out "$OUTDIR" --epochs 50 --lr 1e-4 --resume
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" \
        --task sorting --attn stieltjes --q "$Q" --out "$OUTDIR/analysis"
done
