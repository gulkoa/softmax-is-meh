#!/bin/bash
#SBATCH --job-name=bench-triton-vs-ref-h100
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=36G
#SBATCH --time=01:00:00
#SBATCH --output=results/bench-triton-vs-ref-h100-%j.out
#SBATCH --error=results/bench-triton-vs-ref-h100-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR"

echo "=== H100 bench Triton vs PyTorch-ref (num_iter=3, per-row init) ==="
nvidia-smi

python triton/bench_triton_vs_ref.py \
    --out "results/bench_triton_vs_ref_cardinal_h100.csv"
