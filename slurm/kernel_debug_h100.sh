#!/bin/bash
#SBATCH --job-name=kernel-debug
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-debug-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-debug-%j.err

# Runs a debug script from the kernel-review-fixes branch worktree.
# Usage: sbatch kernel_debug_h100.sh <script.py>

set -euo pipefail
SCRIPT="${1:?usage: sbatch kernel_debug_h100.sh <script.py>}"
cd /users/PAS2402/alexg/softmax/.worktrees/kernel-review-fixes/triton

uv run --project /users/PAS2402/alexg/softmax/softmax-is-meh/triton \
  --no-sync python "${SCRIPT}"

echo "ALL DONE"
