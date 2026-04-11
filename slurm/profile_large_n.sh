#!/bin/bash
#SBATCH --job-name=stieltjes-profile-large-n
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=results/profile-large-n-%j.out
#SBATCH --error=results/profile-large-n-%j.err

set -euo pipefail

REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"
mkdir -p results

export RESULTS_DIR="${REPO_DIR}/results"
python triton/profile_large_n.py
