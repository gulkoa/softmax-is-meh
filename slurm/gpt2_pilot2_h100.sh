#!/bin/bash
#SBATCH --job-name=gpt2-pilot2
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-pilot2-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-pilot2-%j.err

# Improvement #1 pilots: Stieltjes-specific lr and q at 45M/2B tokens
# (softmax-tuned 6e-4 baseline: stj ppl 34.40, sdpa 34.01).
# Usage: sbatch gpt2_pilot2_h100.sh <sdpa|stj> [extra trainer args]

set -euo pipefail

ARM="${1:?usage: sbatch gpt2_pilot2_h100.sh <arm> [args]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "${ARM}" --n-layer 6 --n-head 8 --n-embd 512 \
  --micro-bs 32 --grad-accum 8 --total-tokens 2e9 --warmup 700 \
  --tag pilot2 "$@"

echo "ALL DONE"
