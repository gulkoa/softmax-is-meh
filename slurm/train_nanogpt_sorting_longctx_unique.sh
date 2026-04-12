#!/bin/bash
#SBATCH --job-name=nanogpt-sort-longctx-u
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=results/nanogpt-sort-longctx-u-%j.out
#SBATCH --error=results/nanogpt-sort-longctx-u-%j.err

# Long-context sorting with UNIQUE permutations (max_val = max_arr_len = 256).
# The original longctx sorting had max_val=256 with max_arr_len=1024, making it
# trivial (softmax hit 99.9% in 3 epochs). Using unique values forces real
# attention-based comparison.
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# max_arr_len=256 arrays with max_val=256 (unique values 0-255, sorted permutations)
# seq_len=1024 easily fits 256 input + 1 sep + 256 output + padding
COMMON="--task sorting --seq-len 1024 --max-arr-len 256 --max-val 256 --lr 1e-4 --epochs 30 --batch-size 8 --train-samples 50000 --val-samples 5000 --resume"

# Softmax baseline
OUTDIR="results/sorting_softmax_ctx1024_unique"
mkdir -p "$OUTDIR"
echo "=== Sorting softmax ctx=1024 unique ==="
python nanogpt/train.py $COMMON --attn softmax --out "$OUTDIR"
python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" --task sorting --attn softmax --out "$OUTDIR/analysis" --seq-len 1024 --max-arr-len 256 --max-val 256

# Stieltjes at key q values
for Q in 1.0 2.0 4.0 8.0; do
    OUTDIR="results/sorting_stieltjes_q${Q}_ctx1024_unique"
    mkdir -p "$OUTDIR"
    echo "=== Sorting stieltjes q=$Q ctx=1024 unique ==="
    python nanogpt/train.py $COMMON --attn stieltjes --q "$Q" --out "$OUTDIR"
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" --task sorting --attn stieltjes --q "$Q" --out "$OUTDIR/analysis" --seq-len 1024 --max-arr-len 256 --max-val 256
done
