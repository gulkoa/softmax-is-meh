#!/bin/bash
#SBATCH --job-name=check-triton-blg
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/check-triton-blg-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/check-triton-blg-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python tmp/check_triton_block_lambda_grad.py

echo "DONE"
