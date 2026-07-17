#!/bin/bash
#SBATCH --job-name=gpt2-pilot
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-pilot-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-pilot-%j.err

# GPT-2 project M1 pilot: 6L/8H/512 on FineWeb-Edu shards, 2B tokens.
# Usage: sbatch gpt2_pilot_h100.sh <sdpa|stj>

set -euo pipefail

ARM="${1:?usage: sbatch gpt2_pilot_h100.sh <sdpa|stj>}"

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "${ARM}" --n-layer 6 --n-head 8 --n-embd 512 \
  --micro-bs 32 --grad-accum 8 --total-tokens 2e9 --warmup 700 \
  --tag pilot

echo "ALL DONE"
