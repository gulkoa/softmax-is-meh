#!/bin/bash
#SBATCH --job-name=eval-gpt2
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/eval-gpt2-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/eval-gpt2-%j.err

# GPT-2 project M3: fineweb/wikitext ppl + LAMBADA + completions.
# Usage: sbatch eval_gpt2_h100.sh <ckpt.pt> [...]

set -euo pipefail
cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/eval_gpt2_stieltjes.py "$@"

echo "ALL DONE"
