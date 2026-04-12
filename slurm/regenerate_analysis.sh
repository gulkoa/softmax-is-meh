#!/bin/bash
#SBATCH --job-name=stj-regen-analysis
#SBATCH --partition=gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=results/regen-analysis-%j.out
#SBATCH --error=results/regen-analysis-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
cd "$REPO_DIR"
bash scripts/regenerate_analysis.sh
