#!/bin/bash
#SBATCH --job-name=mqmtar-v7
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-v7-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-v7-%j.err

# v7 diagnostic runs (checkpoint + generation printouts). Default here is
# the lr-probe protocol: 1/4 budget (97,656 steps, warmup 5,000) on the
# 50M set, lr from CLI. Discriminates optimization-limited (higher lr
# breaks the 0.55 floor) from harness/capability-limited (same floor).
# Usage: sbatch mqmtar_h2h_v7_h100.sh <arm> [--nape] [--lr X] [...]

set -euo pipefail

ARM="${1:?usage: sbatch mqmtar_h2h_v7_h100.sh <arm> [flags]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
export MQMTAR_DATA=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar_data/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4

echo "########## MQMTAR v7 diag: arm=${ARM} extra=[$*] ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/mqmtar_headtohead_v7.py \
  --arm "${ARM}" --seed 0 --steps 97656 --warmup 5000 "$@"

echo "ALL DONE"
