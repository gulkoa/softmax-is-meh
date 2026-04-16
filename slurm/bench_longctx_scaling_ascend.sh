#!/bin/bash
# Long-context scaling benchmark: throughput + peak memory vs N (up to 131072).
# Compares Triton Stieltjes, flash SDPA, PyTorch-ref Stieltjes, naive softmax.
#SBATCH --job-name=stj-longctx-bench
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=results/bench-longctx-%j.out
#SBATCH --error=results/bench-longctx-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"
mkdir -p results

export RESULTS_DIR="${REPO_DIR}/results"
python triton/bench_longctx_scaling.py
