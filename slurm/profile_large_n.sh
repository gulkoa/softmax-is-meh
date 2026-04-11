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

# Resolve repo root via marker file
_search="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$_search" != "/" ]; do
    if [ -f "$_search/triton/pyproject.toml" ]; then REPO_DIR="$_search"; break; fi
    _search="$(dirname "$_search")"
done
if [ -z "${REPO_DIR:-}" ]; then echo "ERROR: repo root not found" >&2; exit 1; fi

source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"
mkdir -p results

export RESULTS_DIR="${REPO_DIR}/results"
python triton/profile_large_n.py
