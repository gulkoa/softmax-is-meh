#!/bin/bash
#SBATCH --job-name=mqar-hardstretch
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqar-hardstretch-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqar-hardstretch-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
export PYTHONPATH=/users/PAS2402/alexg/softmax/zoology:/users/PAS2402/alexg/softmax/softmax-is-meh/triton/dev_scripts

uv run --project softmax-is-meh/triton --no-sync python -m zoology.launch \
  softmax-is-meh/triton/dev_scripts/zoology_mqar_hardstretch.py

echo "ALL DONE"
