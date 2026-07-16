#!/bin/bash
#SBATCH --job-name=verify-hf-serving
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/verify-hf-serving-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/verify-hf-serving-%j.err

# Verify the HF trust_remote_code packaging reproduces the native model
# (25 MQMTAR samples through AutoModelForCausalLM.generate on GPU).
# Usage: sbatch verify_hf_serving_h100.sh <hf_repo_dir>

set -euo pipefail
cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/verify_hf_serving.py "$@"

echo "ALL DONE"
