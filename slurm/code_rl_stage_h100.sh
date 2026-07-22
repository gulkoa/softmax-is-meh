#!/bin/bash
#SBATCH --job-name=code-rl-stage
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/code-rl-stage-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/code-rl-stage-%j.err

# Generic runner for the code-RL pipeline stages (code-SFT / pass@k
# eval / GRPO). Plan: thesis/findings/2026-07-20-plan-rl-coding-stilt11.md
# Usage: sbatch [-t HH:MM:SS] code_rl_stage_h100.sh <script.py> [args...]
#   e.g. sbatch -t 01:00:00 code_rl_stage_h100.sh sft_code_stilt.py <ckpt>
#        sbatch -t 03:00:00 code_rl_stage_h100.sh eval_code_stilt.py <ckpt>

set -euo pipefail

SCRIPT="${1:?usage: sbatch code_rl_stage_h100.sh <script.py> [args]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

uv run --project softmax-is-meh/triton --no-sync python \
  "softmax-is-meh/triton/dev_scripts/${SCRIPT}" "$@"

echo "ALL DONE"
