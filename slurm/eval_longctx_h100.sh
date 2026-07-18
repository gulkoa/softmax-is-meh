#!/bin/bash
#SBATCH --job-name=eval-longctx
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/eval-longctx-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/eval-longctx-%j.err

# Long-context NL comparison (context-utilization + beyond-ctx ppl) on
# the trained 124M pair. Usage: sbatch eval_longctx_h100.sh <ckpts...>

set -euo pipefail
cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/eval_longctx_gpt2.py "$@"

echo "ALL DONE"
