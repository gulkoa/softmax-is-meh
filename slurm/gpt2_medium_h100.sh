#!/bin/bash
#SBATCH --job-name=gpt2-medium
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-medium-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-medium-%j.err

# Task #31: GPT-2 medium (355M), 15B tokens, web+math+code mix
# (70% FineWeb-Edu / 15% FineMath-4+ / 15% codeparrot-py).
# Checkpoint/resume: resubmit the same command to continue after a cut.
# Usage: sbatch gpt2_medium_h100.sh <sdpa|stj>

set -euo pipefail

ARM="${1:?usage: sbatch gpt2_medium_h100.sh <sdpa|stj>}"

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7"
MIX="${MIX},math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15"
MIX="${MIX},code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "${ARM}" --n-layer 24 --n-head 16 --n-embd 1024 \
  --micro-bs 8 --grad-accum 64 --total-tokens 15e9 --warmup 2000 \
  --data-mix "${MIX}" --tag medium-mix

echo "ALL DONE"
