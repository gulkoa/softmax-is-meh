#!/bin/bash
#SBATCH --job-name=gpt2-rope
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-rope-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/gpt2-rope-%j.err

# 355M RoPE arm (stilt.1.1 roadmap item 1) — same 15B web+math+code mix
# and scale-learnable recipe as the nope pair; rotary q/k pre-kernel.
# Runs the trainer from the stilt11-trainer BRANCH WORKTREE so main's
# trainer stays frozen while the nope chain is queued. Merge the branch
# and repoint here at the next unfreeze.
# Usage: sbatch gpt2_rope_h100.sh <sdpa|stj> [extra args, last-wins]

set -euo pipefail

ARM="${1:?usage: sbatch gpt2_rope_h100.sh <arm> [args]}"
shift

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7"
MIX="${MIX},math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15"
MIX="${MIX},code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"

uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/stilt11-trainer/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "${ARM}" --n-layer 24 --n-head 16 --n-embd 1024 \
  --micro-bs 8 --grad-accum 64 --total-tokens 15e9 --warmup 2000 \
  --data-mix "${MIX}" --rope --scale-learnable --tag mix "$@"

echo "ALL DONE"
