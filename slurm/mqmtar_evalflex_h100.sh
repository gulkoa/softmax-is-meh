#!/bin/bash
#SBATCH --job-name=mqmtar-flex
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-flex-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-flex-%j.err

# FlexAttention long-context eval (16k/32k/65k) of a softmaxd+NAPE
# checkpoint — extends the softmax row past the dense 8k ceiling into the
# 256x-1024x regime where the published claims live.
# Usage: sbatch mqmtar_evalflex_h100.sh <ckpt>

set -euo pipefail

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/mqmtar_eval_flex.py "$@"

echo "ALL DONE"
