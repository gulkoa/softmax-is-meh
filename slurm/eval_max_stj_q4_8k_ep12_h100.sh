#!/bin/bash
#SBATCH --job-name=eval-max-q4-8k-ep12
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/eval-max-q4-8k-ep12-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/eval-max-q4-8k-ep12-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax/softmax-is-meh
source triton/.venv/bin/activate

OUTDIR=results/max_stieltjes_q4.0_seq8192_cardinal_h100

python nanogpt/eval_accuracy.py \
  --checkpoint "$OUTDIR/model_ep012.pt" \
  --task max \
  --attn stieltjes \
  --q 4.0 \
  --seq-len 8192 \
  --max-arr-len 8184 \
  --max-val 64 \
  --val-samples 500 \
  --batch-size 1 \
  --n-layer 6 \
  --n-head 6 \
  --n-embd 384 \
  --seed 42 \
  --out "$OUTDIR/accuracy_fixed_ep12.json"

echo "ep12 eval done"
