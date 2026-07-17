#!/bin/bash
#SBATCH --job-name=bench-gpt2-shapes
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-gpt2-shapes-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-gpt2-shapes-%j.err

# GPT-2 project M0: attention fwd+bwd throughput at GPT-2-small shapes.

set -euo pipefail
cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/bench_gpt2_shapes.py

echo "ALL DONE"
