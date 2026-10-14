#!/bin/bash
#SBATCH --job-name=passkey
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/passkey-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/passkey-%j.err
# Passkey retrieval (1-8x context) on a shipped model (HF staging dir).
# Usage: sbatch passkey_hf_h100.sh <model dir> [eval_passkey_hf.py args]
set -euo pipefail
MODEL=${1:?model dir}; shift
cd /users/PAS2402/alexg/softmax/softmax-is-meh/triton
export HF_HUB_OFFLINE=1
.venv/bin/python dev_scripts/eval_passkey_hf.py "$MODEL" --trials 20 --alpha 0,0.35 "$@"
echo "ALL DONE"
