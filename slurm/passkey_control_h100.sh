#!/bin/bash
#SBATCH --job-name=passkey-ctrl
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/passkey-ctrl-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/passkey-ctrl-%j.err
# No-question passkey control on the matched-recipe 355M pair (pre-registered
# in thesis/experiments/2026-10-14-passkey-retrieval-bases.md): 1/2/4x, alpha
# 0 and 0.35, 20 trials, failed completions printed.
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh/triton
export HF_HUB_OFFLINE=1
S=/users/PAS2402/alexg/softmax/softmax-is-meh/results
for d in hf_355m_nope_qk_staging hf_355m_nope_qk_softmax_staging; do
  .venv/bin/python dev_scripts/eval_passkey_hf_v2.py $S/$d --template noquestion --lens 1024,2048,4096 --alpha 0,0.35 --trials 20 --show-fails 2
done
echo "ALL DONE"
