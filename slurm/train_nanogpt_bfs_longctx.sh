#!/bin/bash
#SBATCH --job-name=nanogpt-bfs-longctx
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=results/nanogpt-bfs-longctx-%j.out
#SBATCH --error=results/nanogpt-bfs-longctx-%j.err

# Long context BFS: seq_len=4096, larger graphs
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

COMMON="--task bfs --seq-len 4096 --max-arr-len 256 --epochs 30 --batch-size 2 --train-samples 10000 --val-samples 1000"

# Softmax baseline
OUTDIR="results/bfs_softmax_ctx4096"
mkdir -p "$OUTDIR"
echo "=== BFS softmax ctx=4096 ==="
python nanogpt/train.py $COMMON --attn softmax --out "$OUTDIR"
python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" --task bfs --attn softmax --out "$OUTDIR/analysis" --seq-len 4096

# Stieltjes at key q values
for Q in 1.0 2.0 8.0; do
    OUTDIR="results/bfs_stieltjes_q${Q}_ctx4096"
    mkdir -p "$OUTDIR"
    echo "=== BFS stieltjes q=$Q ctx=4096 ==="
    python nanogpt/train.py $COMMON --attn stieltjes --q "$Q" --out "$OUTDIR"
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" --task bfs --attn stieltjes --q "$Q" --out "$OUTDIR/analysis" --seq-len 4096
done
