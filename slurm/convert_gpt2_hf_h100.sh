#!/bin/bash
#SBATCH --job-name=convert-gpt2-hf
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/convert-gpt2-hf-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/convert-gpt2-hf-%j.err

# M4: convert + verify the Stieltjes GPT-2 checkpoint for HF serving.

set -euo pipefail
cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/convert_verify_gpt2_hf.py "$@"

echo "ALL DONE"
