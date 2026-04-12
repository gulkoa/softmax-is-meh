#!/bin/bash
#SBATCH --job-name=nanogpt-max-stj
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/nanogpt-max-stj-%j.out
#SBATCH --error=results/nanogpt-max-stj-%j.err

# All q values including high-q — PyTorch ref for training
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

for Q in 1.0 2.0 4.0 8.0 16.0 32.0 64.0; do
    OUTDIR="results/max_stieltjes_q${Q}_v3"
    mkdir -p "$OUTDIR"
    echo "=== Training max with stieltjes q=$Q (PyTorch ref) ==="
    python nanogpt/train.py --task max --attn stieltjes --q "$Q" --out "$OUTDIR" --epochs 50
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" \
        --task max --attn stieltjes --q "$Q" --out "$OUTDIR/analysis"
done
