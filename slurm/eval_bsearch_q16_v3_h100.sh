#!/bin/bash
#SBATCH --job-name=eval-bsearch-q16-v3
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/eval-bsearch-q16-v3-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/eval-bsearch-q16-v3-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax/softmax-is-meh
source triton/.venv/bin/activate

OUTDIR=results/binary_search_stieltjes_q16.0_v3

python nanogpt/eval_accuracy.py \
  --checkpoint "$OUTDIR/model.pt" \
  --task binary_search \
  --attn stieltjes \
  --q 16.0 \
  --seq-len 128 \
  --max-arr-len 16 \
  --max-val 64 \
  --val-samples 5000 \
  --batch-size 64 \
  --n-layer 6 \
  --n-head 6 \
  --n-embd 384 \
  --seed 42 \
  --out "$OUTDIR/accuracy_fixed.json"

echo "v3 bsearch q=16 eval done"
