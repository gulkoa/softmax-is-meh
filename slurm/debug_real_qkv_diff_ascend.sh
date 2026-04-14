#!/bin/bash
#SBATCH --job-name=debug-real-qkv-diff-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=results/debug-real-qkv-diff-ascend-%j.out
#SBATCH --error=results/debug-real-qkv-diff-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "${REPO_DIR}/triton"
python debug_real_qkv_diff.py
