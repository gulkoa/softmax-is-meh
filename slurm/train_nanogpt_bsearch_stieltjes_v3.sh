#!/bin/bash
#SBATCH --job-name=nanogpt-bsrch-stj-v3
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/nanogpt-bsrch-stj-v3-%j.out
#SBATCH --error=results/nanogpt-bsrch-stj-v3-%j.err

# v3: uses PyTorch reference (stable backward) instead of Triton kernel
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

for Q in 1.0 2.0 4.0 8.0; do
    OUTDIR="results/binary_search_stieltjes_q${Q}_v3"
    mkdir -p "$OUTDIR"
    echo "=== Training binary_search with stieltjes q=$Q (PyTorch ref, lr=3e-4) ==="
    python nanogpt/train.py --task binary_search --attn stieltjes --q "$Q" --out "$OUTDIR" --epochs 50
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" \
        --task binary_search --attn stieltjes --q "$Q" --out "$OUTDIR/analysis"
done
