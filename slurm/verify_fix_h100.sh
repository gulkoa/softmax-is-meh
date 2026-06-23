#!/bin/bash
#SBATCH --job-name=verify-fix
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/verify-fix-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/verify-fix-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python tmp/verify_fix_selftests.py

echo "DONE"
