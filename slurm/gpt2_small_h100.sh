#!/bin/bash
#SBATCH --job-name=gpt2-small
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-small-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-small-%j.err

# GPT-2 project M2: GPT-2 small (124M), 10B FineWeb-Edu tokens.
# Trainer checkpoints every 500 steps and RESUMES automatically —
# resubmit the same command if the job is cut.
# Usage: sbatch gpt2_small_h100.sh <sdpa|stj>

set -euo pipefail

ARM="${1:?usage: sbatch gpt2_small_h100.sh <sdpa|stj>}"

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "${ARM}"

echo "ALL DONE"
