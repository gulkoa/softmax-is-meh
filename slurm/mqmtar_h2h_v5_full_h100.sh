#!/bin/bash
#SBATCH --job-name=mqmtar-full
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-full-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-full-%j.err

# MQMTAR head-to-head at FULL published budget on the v5 trainer
# (= v4 + flash-arm eval n=100 at L>=32k + per-split timing).
# 390,625 steps, warmup 20,000, batch 128, lr 1e-5; 50M-pair set (1 epoch).
# Usage: sbatch --time=HH:00:00 [--dependency=afterok:<datagen>] \
#          mqmtar_h2h_v5_full_h100.sh <arm> [--nape] [--stj-q N]

set -euo pipefail

ARM="${1:?usage: sbatch mqmtar_h2h_v5_full_h100.sh <sdpa|asentmax|asstj> [flags]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
export MQMTAR_DATA=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar_data/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4

echo "########## MQMTAR FULL budget (v5): arm=${ARM} extra=[$*] data=${MQMTAR_DATA} ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/mqmtar_headtohead_v5.py \
  --arm "${ARM}" --seed 0 --steps 390625 --warmup 20000 "$@"

echo "ALL DONE"
