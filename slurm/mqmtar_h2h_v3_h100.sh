#!/bin/bash
#SBATCH --job-name=mqmtar-h2h-v3
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-h2h-v3-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar-h2h-v3-%j.err

# MQMTAR head-to-head, V3 trainer (= V2 + cuDNN SDPA disabled; see
# mqmtar_headtohead_v3.py header). Used for the sdpa arm after job
# 12345902 hit "cuDNN Frontend: No valid execution plans" on odd seq lens.
# Usage: sbatch mqmtar_h2h_v3_h100.sh <arm> [--nape] [--stj-q N]

set -euo pipefail

ARM="${1:?usage: sbatch mqmtar_h2h_v3_h100.sh <sdpa|asentmax|asstj> [flags]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

echo "########## MQMTAR head-to-head v3: arm=${ARM} extra=[$*] ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/mqmtar_headtohead_v3.py \
  --arm "${ARM}" --seed 0 "$@"

echo "ALL DONE"
