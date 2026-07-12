#!/bin/bash
#SBATCH --job-name=bench-flashn-jack
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-flashn-jack-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/bench-flashn-jack-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/bench_flashn_vs_jack.py

echo "DONE"
