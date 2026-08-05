#!/bin/bash
#SBATCH --job-name=kernel-tests
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-tests-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-tests-%j.err

# GPU validation of the kernel-review-fixes branch (task #39): forward,
# backward, and the new non-contiguous B>1 regression tests, run FROM
# THE BRANCH WORKTREE (main's kernel stays frozen for the live chains).
# The file's __main__ runs benchmarks after tests pass; a 1h TIMEOUT
# mid-benchmark is fine — the test verdict lines are what this job is
# for. Merge gate: all three suites must print PASS.

set -euo pipefail
cd /users/PAS2402/alexg/softmax/.worktrees/kernel-review-fixes/triton

uv run --project /users/PAS2402/alexg/softmax/softmax-is-meh/triton \
  --no-sync python stieltjes_flash_attn.py

echo "ALL DONE"
