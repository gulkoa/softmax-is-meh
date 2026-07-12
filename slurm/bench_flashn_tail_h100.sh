#!/bin/bash
#SBATCH --job-name=bench-flashn-tail
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-flashn-tail-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-flashn-tail-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

# Tail of the sweep: fp16 kernel only at large N (fp32-IEEE is the slow
# validation mode and jack already OOM'd at 32k in job 12260585).
BENCH_NS="32768,65536,131072" \
BENCH_PROVIDERS="flashn-fp16" \
BENCH_OUT="/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench_flashn_vs_jack_tail.json" \
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/bench_flashn_vs_jack.py

echo "DONE"
