#!/bin/bash
#SBATCH --job-name=check-triton-correctness-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=results/check-triton-correctness-ascend-%j.out
#SBATCH --error=results/check-triton-correctness-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "${REPO_DIR}/triton"
python check_triton_correctness.py
