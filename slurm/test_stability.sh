#!/bin/bash
#SBATCH --job-name=stj-stability-test
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=results/stability-test-%j.out
#SBATCH --error=results/stability-test-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"
mkdir -p results

python nanogpt/test_stability.py
