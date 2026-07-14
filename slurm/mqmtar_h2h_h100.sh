#!/bin/bash
#SBATCH --job-name=mqmtar-h2h
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-h2h-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-h2h-%j.err

# MQMTAR head-to-head on the ASEntmax protocol (1/16 budget).
# Usage: sbatch mqmtar_h2h_h100.sh <arm>   with arm in {sdpa, asentmax, asstj}
# Data: results/mqmtar_data/3M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4
# (generated with deep-spin/asentmax scripts/generate_data.py verbatim).

set -euo pipefail

ARM="${1:?usage: sbatch mqmtar_h2h_h100.sh <sdpa|asentmax|asstj>}"

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

echo "########## MQMTAR head-to-head: arm=${ARM} ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/mqmtar_headtohead.py \
  --arm "${ARM}" --seed 0

echo "ALL DONE"
