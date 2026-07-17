#!/bin/bash
#SBATCH --job-name=kernel-halley-val
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-halley-val-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-halley-val-%j.err

# Halley-solver validation (convergence vs bisection-80 ground truth,
# gradient parity, latency) + regression of default modes.

set -euo pipefail
cd /users/PAS2402/alexg/softmax

echo "########## 1. Halley solver: convergence + parity + speed ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/test_halley_solver.py

echo ""
echo "########## 2. regression: normalized suite (default NR path) ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/test_triton_normalized.py

echo ""
echo "########## 3. regression: normalized-IFT suite ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/test_ift_norm_backward.py

echo "ALL DONE"
