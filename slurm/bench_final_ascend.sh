#!/bin/bash
#SBATCH --job-name=stj-bench-final-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=results/bench-final-ascend-%j.out
#SBATCH --error=results/bench-final-ascend-%j.err

set -euo pipefail

REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"
mkdir -p results

# NOTE: Ascend has A100 GPUs, not H100 like Cardinal. Throughput numbers will
# differ from Cardinal runs — do not merge into a single CSV.
export RESULTS_DIR="${REPO_DIR}/results"
python triton/bench_final.py
mv "${RESULTS_DIR}/final_benchmark.csv" "${RESULTS_DIR}/final_benchmark_ascend_a100.csv"
