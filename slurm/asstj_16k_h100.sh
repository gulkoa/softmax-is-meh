#!/bin/bash
#SBATCH --job-name=asstj-16k
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/asstj-16k-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/asstj-16k-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

SWEEP_DEMBS="128,256" SWEEP_QORDERS="4,16" \
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/asstj_16k_stretch.py

echo "ALL DONE"
