#!/bin/bash
#SBATCH --job-name=nanogpt-needle-lctx
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=results/nanogpt-needle-lctx-%j.out
#SBATCH --error=results/nanogpt-needle-lctx-%j.err

# Long-context NEEDLE task: find the one "needle" (value 128-254) in a sea of
# "background" tokens (value 0-127). Designed to genuinely test long-range attention.
# seq_len=8192 with max_arr_len=8000 means actual data spans the full context.
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# seq_len=8192 is the max feasible with PyTorch ref attention (N×N matrix) on H100 80GB
# at batch_size=1. Attention matrix at this size: 1×6×8192²×4 = 1.6GB per layer.
COMMON="--task needle --seq-len 8192 --max-arr-len 8000 --lr 1e-4 --epochs 30 --batch-size 1 --train-samples 20000 --val-samples 2000 --resume"

# Softmax baseline
OUTDIR="results/needle_softmax_ctx8192"
mkdir -p "$OUTDIR"
echo "=== Needle softmax ctx=8192 ==="
python nanogpt/train.py $COMMON --attn softmax --out "$OUTDIR"

# Stieltjes at full q range — needle task should especially benefit from high q
# (needle at a specific position = sharp attention target)
for Q in 1.0 2.0 8.0 32.0 64.0; do
    OUTDIR="results/needle_stieltjes_q${Q}_ctx8192"
    mkdir -p "$OUTDIR"
    echo "=== Needle stieltjes q=$Q ctx=8192 ==="
    python nanogpt/train.py $COMMON --attn stieltjes --q "$Q" --out "$OUTDIR"
done
