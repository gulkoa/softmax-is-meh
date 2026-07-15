#!/bin/bash
#SBATCH --job-name=mqmtar-full6
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-full6-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-full6-%j.err

# MQMTAR full published budget, v6 trainer (dense NAPE-capable arms after
# the PE pivot — see mqmtar_headtohead_v6.py header). 390,625 steps,
# warmup 20,000, lr 1e-5, 50M-pair set.
# Usage: sbatch --time=HH:00:00 mqmtar_h2h_v6_full_h100.sh <arm> [--nape] [--stj-q N]

set -euo pipefail

ARM="${1:?usage: sbatch mqmtar_h2h_v6_full_h100.sh <arm> [flags]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
export MQMTAR_DATA=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar_data/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4

echo "########## MQMTAR FULL budget (v6): arm=${ARM} extra=[$*] ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/mqmtar_headtohead_v6.py \
  --arm "${ARM}" --seed 0 --steps 390625 --warmup 20000 "$@"

echo "ALL DONE"
