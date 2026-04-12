#!/bin/bash
#SBATCH --job-name=nanogpt-max-stj-cont
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=results/nanogpt-max-stj-cont-%j.out
#SBATCH --error=results/nanogpt-max-stj-cont-%j.err

# Continue max stieltjes from where it timed out (q=32 was partial, q=64 not started)
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# q=32 had partial data (ep19 of 50) — resume from checkpoint
for Q in 32.0 64.0; do
    OUTDIR="results/max_stieltjes_q${Q}_v3"
    mkdir -p "$OUTDIR"
    echo "=== Max stieltjes q=$Q (resume if checkpoint) ==="
    python nanogpt/train.py --task max --attn stieltjes --q "$Q" --out "$OUTDIR" --epochs 50 --resume
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" \
        --task max --attn stieltjes --q "$Q" --out "$OUTDIR/analysis"
done
