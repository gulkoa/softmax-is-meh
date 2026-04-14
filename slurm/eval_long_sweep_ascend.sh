#!/bin/bash
#SBATCH --job-name=eval-long-sweep-ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=results/eval-long-sweep-ascend-%j.out
#SBATCH --error=results/eval-long-sweep-ascend-%j.err

set -euo pipefail
REPO_DIR="/users/PAS2402/alexg/softmax/softmax-is-meh"
cd "$REPO_DIR"
bash scripts/eval_long_sweep_ascend.sh
