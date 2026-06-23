#!/bin/bash
#SBATCH --job-name=quantify-keycol
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/quantify-keycol-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/quantify-keycol-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python tmp/quantify_padded_keycol_bug.py

echo "DONE"
