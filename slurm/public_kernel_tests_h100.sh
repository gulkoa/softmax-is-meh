#!/bin/bash
#SBATCH --job-name=pub-kernel-tests
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/pub-kernel-tests-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/pub-kernel-tests-%j.err

# GPU test of the PUBLIC stieltjes-triton repo's kernel suite with the
# mirrored H_eff fix applied (task #39) — push gate for the public fix.

set -euo pipefail
cd /users/PAS2402/alexg/softmax/stieltjes-triton

uv run --project /users/PAS2402/alexg/softmax/softmax-is-meh/triton \
  --no-sync python -m pytest tests/test_kernel.py -v

echo "ALL DONE"
