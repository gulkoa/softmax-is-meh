#!/bin/bash
#SBATCH --job-name=stieltjes-lambda-init
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=results/lambda-init-%j.out
#SBATCH --error=results/lambda-init-%j.err

set -euo pipefail

REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"
mkdir -p results

export RESULTS_DIR="${REPO_DIR}/results"
python triton/bench_lambda_init.py
