#!/bin/bash
#SBATCH --job-name=mqmtar-v8
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-v8-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-v8-%j.err

# v8 probes (bounded-beta / q / lr for the asstjd stability problem).
# Default 1/4 budget; pass --steps/--warmup to override (submit with a
# larger --time for full budget).
# Usage: sbatch [--time=..] mqmtar_h2h_v8_h100.sh <arm> [flags]

set -euo pipefail

ARM="${1:?usage: sbatch mqmtar_h2h_v8_h100.sh <arm> [flags]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
export MQMTAR_DATA=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar_data/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4

echo "########## MQMTAR v8: arm=${ARM} extra=[$*] ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/mqmtar_headtohead_v8.py \
  --arm "${ARM}" --seed 0 --steps 97656 --warmup 5000 "$@"

echo "ALL DONE"
