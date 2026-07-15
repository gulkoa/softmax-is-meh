#!/bin/bash
#SBATCH --job-name=bench-bh-scaling
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-bh-scaling-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-bh-scaling-%j.err

# FLASHn BH x D x N x num_iter throughput matrix vs SDPA (task #26:
# MQMTAR eval forensics found ~80x at BH=1024/D=32 long-N where sweep
# count explains ~10x). Short job — backfills around the full-budget runs.

set -euo pipefail
cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/bench_flashn_bh_scaling.py

echo "ALL DONE"
