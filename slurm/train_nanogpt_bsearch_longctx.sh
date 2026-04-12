#!/bin/bash
#SBATCH --job-name=nanogpt-bsrch-longctx
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=results/nanogpt-bsrch-longctx-%j.out
#SBATCH --error=results/nanogpt-bsrch-longctx-%j.err

# Long-context binary search. This is the RIGHT long-context experiment:
# - Non-trivial (requires precise positional attention)
# - Unique values (no ambiguity)
# - Tests whether q=8 breakthrough on short context scales to long context
# - If Stieltjes q=8 still wins at long context, we have the full story
set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

# Array of unique values, target to find, output is index
# seq_len=1024 fits 256 array + separator + target + separator + index
COMMON="--task binary_search --seq-len 1024 --max-arr-len 256 --max-val 256 --lr 1e-4 --epochs 30 --batch-size 8 --train-samples 50000 --val-samples 5000 --resume"

# Softmax baseline
OUTDIR="results/binary_search_softmax_ctx1024"
mkdir -p "$OUTDIR"
echo "=== Binary search softmax ctx=1024 ==="
python nanogpt/train.py $COMMON --attn softmax --out "$OUTDIR"
python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" --task binary_search --attn softmax --out "$OUTDIR/analysis" --seq-len 1024 --max-arr-len 256 --max-val 256

# Stieltjes — focus on q=8 (the breakthrough config) plus bracketing values
for Q in 1.0 4.0 8.0 16.0; do
    OUTDIR="results/binary_search_stieltjes_q${Q}_ctx1024"
    mkdir -p "$OUTDIR"
    echo "=== Binary search stieltjes q=$Q ctx=1024 ==="
    python nanogpt/train.py $COMMON --attn stieltjes --q "$Q" --out "$OUTDIR"
    python nanogpt/analyze.py --checkpoint "$OUTDIR/model.pt" --task binary_search --attn stieltjes --q "$Q" --out "$OUTDIR/analysis" --seq-len 1024 --max-arr-len 256 --max-val 256
done
