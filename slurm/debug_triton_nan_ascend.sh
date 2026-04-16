#!/bin/bash
#SBATCH --job-name=debug-triton-nan-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=results/debug-triton-nan-ascend-%j.out
#SBATCH --error=results/debug-triton-nan-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
source "${REPO_DIR}/triton/.venv/bin/activate"
cd "$REPO_DIR/triton"

echo "### bf16 causal q=4 N=512 (matches real training) ###"
python debug_triton_nan_bwd.py --dtype bf16 --causal --q 4.0 --N 512 --steps 20

echo "### bf16 causal q=8 N=512 ###"
python debug_triton_nan_bwd.py --dtype bf16 --causal --q 8.0 --N 512 --steps 20

echo "### bf16 causal q=16 N=512 ###"
python debug_triton_nan_bwd.py --dtype bf16 --causal --q 16.0 --N 512 --steps 20

echo "### bf16 causal q=64 N=512 ###"
python debug_triton_nan_bwd.py --dtype bf16 --causal --q 64.0 --N 512 --steps 20

echo "### bf16 causal q=4 N=2048 (matches seq=2048 sweep) ###"
python debug_triton_nan_bwd.py --dtype bf16 --causal --q 4.0 --N 2048 --steps 10

echo "### fp32 causal q=4 N=512 (control) ###"
python debug_triton_nan_bwd.py --dtype fp32 --causal --q 4.0 --N 512 --steps 20
