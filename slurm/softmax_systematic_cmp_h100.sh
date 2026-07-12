#!/bin/bash
#SBATCH --job-name=softmax-syscmp
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/softmax-syscmp-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/softmax-syscmp-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

echo "########## AXIS 2: speed vs SDPA at high context (num_iter=8) ##########"
BENCH_NUM_ITER=8 \
BENCH_NS="16384,32768,65536,131072" \
BENCH_PROVIDERS="flashn-fp16,sdpa-fp16" \
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/bench_causal_dtypes.py \
  || echo ">>> speed axis reported failures; continuing"

echo ""
echo "########## AXIS 1: task quality vs softmax to 512x train length ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/maxretr_vs_softmax.py

echo "ALL DONE"
