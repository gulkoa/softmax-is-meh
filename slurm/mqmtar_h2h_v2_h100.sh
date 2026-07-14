#!/bin/bash
#SBATCH --job-name=mqmtar-h2h-v2
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-h2h-v2-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-h2h-v2-%j.err

# MQMTAR head-to-head, V2 faithful port (see mqmtar_headtohead_v2.py header).
# Usage: sbatch mqmtar_h2h_v2_h100.sh <arm> [--nape]
#   arms: sdpa | asentmax | asstj;  published anchor: asentmax --nape

set -euo pipefail

ARM="${1:?usage: sbatch mqmtar_h2h_v2_h100.sh <sdpa|asentmax|asstj> [--nape]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

echo "########## MQMTAR head-to-head v2: arm=${ARM} extra=[$*] ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/mqmtar_headtohead_v2.py \
  --arm "${ARM}" --seed 0 "$@"

echo "ALL DONE"
