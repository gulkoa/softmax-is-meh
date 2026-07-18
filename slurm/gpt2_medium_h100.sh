#!/bin/bash
#SBATCH --job-name=gpt2-medium
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-medium-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-medium-%j.err

# Scale-up: GPT-2 medium (355M), 25B tokens, MIXED corpus (web+math+code).
# Trainer resumes from its checkpoint automatically — resubmit this same
# command when a chunk hits walltime, until "DONE".
# Usage: sbatch gpt2_medium_h100.sh <sdpa|stj>

set -euo pipefail

ARM="${1:?usage: sbatch gpt2_medium_h100.sh <sdpa|stj>}"

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
export FW_DIRS="/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:/fs/scratch/PAS2836/alexg/finemath_4plus:/fs/scratch/PAS2836/alexg/stackedu_python"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "${ARM}" --n-layer 24 --n-head 16 --n-embd 1024 \
  --micro-bs 16 --grad-accum 32 --total-tokens 25e9 \
  --lr 3e-4 --warmup 2000 --tag medium-mix

echo "ALL DONE"
