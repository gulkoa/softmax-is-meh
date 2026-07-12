#!/bin/bash
#SBATCH --job-name=bench-flashn-iter8
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-flashn-iter8-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-flashn-iter8-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

# Head-to-head at the new num_iter=8 default (converged for q<=16 per job
# 12312766). Skips fp32-IEEE (validation-only mode, walltime-prohibitive).
BENCH_NUM_ITER=8 \
BENCH_PROVIDERS="jack-stj,flashn-fp16" \
BENCH_NS="1024,2048,4096,8192,16384,32768,65536,131072" \
BENCH_OUT="/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench_flashn_vs_jack_iter8.json" \
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/bench_flashn_vs_jack.py

echo "DONE"
