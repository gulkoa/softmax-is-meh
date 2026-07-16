#!/bin/bash
#SBATCH --job-name=kernel-iftnorm-val
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-iftnorm-val-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-iftnorm-val-%j.err

# Validate the kernel's normalized-IFT backward (ift_grad=True): vs dense
# IFT autograd, plus regression of the untouched modes (normalized suite).

set -euo pipefail
cd /users/PAS2402/alexg/softmax

echo "########## 1. normalized-IFT backward vs dense-IFT autograd ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/test_ift_norm_backward.py

echo ""
echo "########## 2. regression: normalized suite (detached modes) ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/test_triton_normalized.py

echo "ALL DONE"
