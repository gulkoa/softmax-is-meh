#!/bin/bash
#SBATCH --job-name=stieltjes-flash-attn
#SBATCH --partition=gpu
#SBATCH --account=ACCOUNT         # <-- REPLACE with your OSC project code (e.g. PAS1234)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err

# ---------------------------------------------------------------------------
# Stieltjes Flash Attention — OSC Cardinal (H100) batch script
#
# Usage:
#   sbatch slurm/run_cardinal.sh                   # default: correctness + benchmark
#   sbatch slurm/run_cardinal.sh --test-only        # correctness tests only
#   sbatch slurm/run_cardinal.sh --bench-only       # benchmarks only
#   sbatch slurm/run_cardinal.sh --all-kernels      # run all triton kernels
#
# To request more GPUs (e.g. for multi-GPU benchmarks):
#   sbatch --gpus-per-node=4 slurm/run_cardinal.sh
# ---------------------------------------------------------------------------

set -euo pipefail

echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPUs:          $SLURM_GPUS_ON_NODE"
echo "Start time:    $(date)"
echo "=========================================="

# --- Environment setup ---
module load pytorch/2.8.0

# Verify GPU access
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}  ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
import triton
print(f'Triton {triton.__version__}')
"

# --- Navigate to repo ---
cd "$SLURM_SUBMIT_DIR"
mkdir -p results

# --- Parse arguments ---
MODE="${1:-default}"

run_tests() {
    echo ""
    echo "====== Stieltjes Flash Attention: Correctness Tests ======"
    python3 -c "
import sys; sys.path.insert(0, 'triton')
from stieltjes_flash_attn import test_forward_correctness, test_backward_correctness
fwd = test_forward_correctness()
bwd = test_backward_correctness()
if not (fwd and bwd):
    sys.exit(1)
"
}

run_benchmarks() {
    echo ""
    echo "====== Stieltjes Flash Attention: Benchmarks ======"
    python3 -c "
import sys, os; sys.path.insert(0, os.path.abspath('triton'))
os.chdir('results')
from stieltjes_flash_attn import benchmark
benchmark()
"
}

run_all_kernels() {
    echo ""
    echo "====== Stieltjes kernel (standalone) ======"
    python3 triton/stieltjes.py

    echo ""
    echo "====== Stieltjes Flash Attention ======"
    python3 triton/stieltjes_flash_attn.py

    echo ""
    echo "====== Stieltjes Benchmarks ======"
    cd triton && python3 bench_stieltjes.py && cd ..

    echo ""
    echo "====== Convergence Benchmarks ======"
    cd triton && python3 bench_convergence.py && cd ..
}

case "$MODE" in
    --test-only)
        run_tests
        ;;
    --bench-only)
        run_benchmarks
        ;;
    --all-kernels)
        run_all_kernels
        ;;
    *)
        run_tests
        run_benchmarks
        ;;
esac

echo ""
echo "=========================================="
echo "Finished at:   $(date)"
echo "=========================================="
