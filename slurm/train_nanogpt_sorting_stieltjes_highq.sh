#!/bin/bash
#SBATCH --job-name=nanogpt-sort-stj-hq
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/nanogpt-sort-stj-hq-%j.out
#SBATCH --error=results/nanogpt-sort-stj-hq-%j.err

# High q values (16, 32, 64) — PyTorch ref for training
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

for Q in 16.0 32.0 64.0; do
    OUTDIR="results/sorting_stieltjes_q${Q}_v3"
    mkdir -p "$OUTDIR"
    echo "=== Training sorting with stieltjes q=$Q (PyTorch ref) ==="
    python nanogpt/train.py --task sorting --attn stieltjes --q "$Q" --out "$OUTDIR" --epochs 50
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" \
        --task sorting --attn stieltjes --q "$Q" --out "$OUTDIR/analysis"
done
