#!/bin/bash
#SBATCH --job-name=nanogpt-sort-longctx
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=results/nanogpt-sort-longctx-%j.out
#SBATCH --error=results/nanogpt-sort-longctx-%j.err

# Long context experiments: seq_len=4096, max_arr_len=1024
# Tests whether algebraic tails help with distant attention
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

COMMON="--task sorting --seq-len 4096 --max-arr-len 1024 --max-val 256 --lr 1e-4 --epochs 30 --batch-size 2 --train-samples 50000 --val-samples 5000"

# Softmax baseline
OUTDIR="results/sorting_softmax_ctx4096"
mkdir -p "$OUTDIR"
echo "=== Sorting softmax ctx=4096 ==="
python nanogpt/train.py $COMMON --attn softmax --out "$OUTDIR"
python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" --task sorting --attn softmax --out "$OUTDIR/analysis" --seq-len 4096

# Stieltjes at key q values
for Q in 1.0 2.0 8.0; do
    OUTDIR="results/sorting_stieltjes_q${Q}_ctx4096"
    mkdir -p "$OUTDIR"
    echo "=== Sorting stieltjes q=$Q ctx=4096 ==="
    python nanogpt/train.py $COMMON --attn stieltjes --q "$Q" --out "$OUTDIR"
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" --task sorting --attn stieltjes --q "$Q" --out "$OUTDIR/analysis" --seq-len 4096
done
