#!/bin/bash
#SBATCH --job-name=triton-vs-ref-seq
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=results/triton-vs-ref-seq-%j.out
#SBATCH --error=results/triton-vs-ref-seq-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

echo "=== Existing correctness check ==="
python triton/check_triton_correctness.py
echo ""
echo "=== Seq-sweep correctness check ==="
python triton/check_triton_seq_sweep.py
