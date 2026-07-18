#!/bin/bash
#SBATCH --job-name=prepare-mathcode
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/prepare-mathcode-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/prepare-mathcode-%j.err

# Task #31: 4B math (FineMath-4+) + 4B code (codeparrot-clean py) tokens.
# Usage: sbatch prepare_mathcode_cpu.sh <math|code>

set -euo pipefail
cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/prepare_mathcode.py "$1"

echo "ALL DONE"
