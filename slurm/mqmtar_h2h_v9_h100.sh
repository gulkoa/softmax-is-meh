#!/bin/bash
#SBATCH --job-name=mqmtar-v9
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-v9-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-v9-%j.err

# v9 stability-study runs (instrumentation + freeze/lr-mult flags).
# Default 1/4 budget; override with --steps/--warmup.
# Usage: sbatch mqmtar_h2h_v9_h100.sh <arm> [flags]

set -euo pipefail

ARM="${1:?usage: sbatch mqmtar_h2h_v9_h100.sh <arm> [flags]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
export MQMTAR_DATA=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar_data/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4

echo "########## MQMTAR v9 stability: arm=${ARM} extra=[$*] ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/mqmtar_headtohead_v9.py \
  --arm "${ARM}" --seed 0 --steps 97656 --warmup 5000 "$@"

echo "ALL DONE"
