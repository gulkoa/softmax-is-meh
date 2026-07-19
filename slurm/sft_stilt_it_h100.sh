#!/bin/bash
#SBATCH --job-name=sft-stilt-it
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/sft-stilt-it-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/sft-stilt-it-%j.err

# stilt -it SFT (smol-smoltalk, assistant-only loss).
# Usage: sbatch sft_stilt_it_h100.sh <base_ckpt.pt>

set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/sft_stilt_it.py "$@"

echo "ALL DONE"
