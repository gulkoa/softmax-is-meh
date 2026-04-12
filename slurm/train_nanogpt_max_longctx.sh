#!/bin/bash
#SBATCH --job-name=nanogpt-max-longctx
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=results/nanogpt-max-longctx-%j.out
#SBATCH --error=results/nanogpt-max-longctx-%j.err

# Long context max/argmax: seq_len=4096, large arrays
# This task should benefit from sharp attention (high q) at long range
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

COMMON="--task max --seq-len 4096 --max-arr-len 1024 --max-val 256 --lr 1e-4 --epochs 30 --batch-size 2 --train-samples 50000 --val-samples 5000"

# Softmax baseline
OUTDIR="results/max_softmax_ctx4096"
mkdir -p "$OUTDIR"
echo "=== Max softmax ctx=4096 ==="
python nanogpt/train.py $COMMON --attn softmax --out "$OUTDIR"

# Stieltjes at key q values
for Q in 1.0 2.0 8.0 32.0; do
    OUTDIR="results/max_stieltjes_q${Q}_ctx4096"
    mkdir -p "$OUTDIR"
    echo "=== Max stieltjes q=$Q ctx=4096 ==="
    python nanogpt/train.py $COMMON --attn stieltjes --q "$Q" --out "$OUTDIR"
done
