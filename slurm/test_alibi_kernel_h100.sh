#!/bin/bash
#SBATCH --job-name=test-alibi-k
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/test-alibi-kernel-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/test-alibi-kernel-%j.err
# GPU tests of the ALiBi bias path in the stjalibi worktree's Stieltjes kernel.
set -euo pipefail
cd /users/PAS2402/alexg/softmax/.worktrees/stjalibi/triton
/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python test_alibi_kernel.py
echo "ALL DONE"
