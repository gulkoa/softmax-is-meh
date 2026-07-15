#!/bin/bash
#SBATCH --job-name=mqmtar-evalckpt
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-evalckpt-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-evalckpt-%j.err

# Recover the long dense splits (4096/8192) that 4h probe walltimes cut
# off, from v7 checkpoints (saved before the eval loop).
# Usage: sbatch [--dependency=afterany:<job>] mqmtar_evalckpt_h100.sh <ckpt> [...]

set -euo pipefail

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/mqmtar_eval_ckpt.py "$@" --lengths 4096 8192

echo "ALL DONE"
