#!/bin/bash
#SBATCH --job-name=nanogpt-bfs-stieltjes
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/nanogpt-bfs-stieltjes-%j.out
#SBATCH --error=results/nanogpt-bfs-stieltjes-%j.err

set -euo pipefail
_search="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$_search" != "/" ]; do
    if [ -f "$_search/triton/pyproject.toml" ]; then REPO_DIR="$_search"; break; fi
    _search="$(dirname "$_search")"
done
if [ -z "${REPO_DIR:-}" ]; then echo "ERROR: repo root not found" >&2; exit 1; fi

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

for Q in 1.0 2.0 4.0 8.0; do
    OUTDIR="results/bfs_stieltjes_q${Q}"
    mkdir -p "$OUTDIR"
    echo "=== Training bfs with stieltjes q=$Q ==="
    python nanogpt/train.py --task bfs --attn stieltjes --q "$Q" --out "$OUTDIR" --epochs 50
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" \
        --task bfs --attn stieltjes --q "$Q" --out "$OUTDIR/analysis"
done
