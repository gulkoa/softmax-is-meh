#!/bin/bash
#SBATCH --job-name=grpo-code
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/grpo-code-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/grpo-code-%j.err

# Code-RL harness (probe or GRPO). Plan:
# thesis/findings/2026-07-20-plan-rl-coding-stilt11.md
# Usage: sbatch grpo_code_h100.sh <ckpt.pt> [--probe|--solvable set.json ...]

set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/grpo_code_stilt.py "$@"

echo "ALL DONE"
