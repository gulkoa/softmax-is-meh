#!/bin/bash
#SBATCH --job-name=gpt2-large
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-large-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-large-%j.err

# 774M (GPT-2 large: 36L/20H/d1280), 30B tokens, web(100BT)+math+code
# 65/20/15. lr guidance from the 355M pair: sdpa diverged at 6e-4 there;
# stj did not. Defaults here: pass --lr per arm (stj 4e-4, sdpa 2e-4
# recommended). ~14 x 16h chunks per arm; chain with
# --dependency=afterany and identical args to resume.
# Usage: sbatch gpt2_large_h100.sh <sdpa|stj> --lr <lr> [extra args]

set -euo pipefail

ARM="${1:?usage: sbatch gpt2_large_h100.sh <arm> [args]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_100bt:0.65"
MIX="${MIX},math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.20"
MIX="${MIX},code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "${ARM}" --n-layer 36 --n-head 20 --n-embd 1280 \
  --micro-bs 4 --grad-accum 128 --total-tokens 30e9 --warmup 2000 \
  --data-mix "${MIX}" --tag large-mix "$@"

echo "ALL DONE"
