#!/bin/bash
#SBATCH --job-name=prof-split
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/profile-step-split-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/profile-step-split-%j.err
# Where a training step's GPU time goes per mapping (record
# thesis/experiments/2026-10-16-profile-step-split.md). 355M P2 config first,
# then the 124M shape. Measurement only: imports the main trainer's model
# read-only; no checkpoints, no wandb.
set -uo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
PY=/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python
S=triton/dev_scripts/profile_step_split.py
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv
echo "== 355M (24 x 16 x 1024, micro-batch 8)"
$PY $S --layers 24 --heads 16 --embd 1024 --micro-bs 8 --grad-accum 64 \
    --json results/profile_step_split_355m_${SLURM_JOB_ID}.json
echo "== 124M (12 x 12 x 768, micro-batch 16)"
$PY $S --layers 12 --heads 12 --embd 768 --micro-bs 16 --grad-accum 32 \
    --json results/profile_step_split_124m_${SLURM_JOB_ID}.json
echo "ALL DONE"
