#!/bin/bash
#SBATCH --job-name=gpt2-medium2
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-medium2-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-medium2-%j.err

# 355M web+math+code, args forwarded (lr overridable). The sdpa arm
# diverged at 6e-4 post-resume (job 12428687, loss 1.9 -> 6.6); classic
# GPT-2-medium lr is 3e-4. The stj arm survives 6e-4 (noted).
# Usage: sbatch gpt2_medium2_h100.sh <sdpa|stj> [extra args]

set -euo pipefail

ARM="${1:?usage: sbatch gpt2_medium2_h100.sh <arm> [args]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7"
MIX="${MIX},math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15"
MIX="${MIX},code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "${ARM}" --n-layer 24 --n-head 16 --n-embd 1024 \
  --micro-bs 8 --grad-accum 64 --total-tokens 15e9 --warmup 2000 \
  --data-mix "${MIX}" --tag medium-mix "$@"

echo "ALL DONE"
