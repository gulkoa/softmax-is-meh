#!/bin/bash
#SBATCH --job-name=prepare-data2
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/prepare-data2-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/prepare-data2-%j.err

# Task #31 774M data: long-walltime prep for big sources (web100 = 21B
# FineWeb-Edu sample-100BT). Small ext tags can use prepare_mathcode_cpu.sh.
# Usage: sbatch prepare_data2_cpu.sh <web100|math-ext|code-ext>

set -euo pipefail
cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/prepare_mathcode.py "$1"

echo "ALL DONE"
