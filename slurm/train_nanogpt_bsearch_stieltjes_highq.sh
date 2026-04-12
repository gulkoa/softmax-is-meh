#!/bin/bash
#SBATCH --job-name=nanogpt-bsrch-stj-hq
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/nanogpt-bsrch-stj-hq-%j.out
#SBATCH --error=results/nanogpt-bsrch-stj-hq-%j.err

# Binary search high-q — completes the q-sensitivity table.
# Given sort/BFS high-q failure, may also fail. But if q=8 solves binary search at
# 99.5%, maybe q=16,32,64 does even better, or shows the failure mode is task-
# dependent. Either way, fills a gap.
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

for Q in 16.0 32.0 64.0; do
    OUTDIR="results/binary_search_stieltjes_q${Q}_v3"
    mkdir -p "$OUTDIR"
    echo "=== Training binary_search with stieltjes q=$Q ==="
    python nanogpt/train.py --task binary_search --attn stieltjes --q "$Q" --out "$OUTDIR" --epochs 50 --resume
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" \
        --task binary_search --attn stieltjes --q "$Q" --out "$OUTDIR/analysis"
done
