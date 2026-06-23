#!/bin/bash
#SBATCH --job-name=nan-truth-table
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/nan-truth-table-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/nan-truth-table-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python tmp/nan_truth_table_gpu.py

echo "DONE"
