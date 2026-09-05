#!/bin/bash
#SBATCH --job-name=nope-qk-stj
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/nope-qk-stj-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/nope-qk-stj-%j.err
# 355M SYMMETRIC RERUN (roadmap open-queue item 1; pre-registration
# thesis/findings/2026-09-04-plan-symmetric-355m-stj-qk.md): the
# Stieltjes counterpart of the fair twin. Byte-identical to
# nope_qk_twin_h100.sh except --attn stj (q=4 default), i.e. the
# original stj-nope arm (12.02 @6e-4, cap recipe) + ONE change:
# --qk-norm. Completes the matched-recipe 355M comparison against
# sdpa-nope+qk (11.90). NO --modern, NO --compile, cap 15 as the twin.
# USER-GATED: submit only via submit_nope_qk_stj_20260904.sh after an
# explicit go. Self-resuming; FINAL-guarded chunks no-op.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7"
MIX="${MIX},math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15"
MIX="${MIX},code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn stj --n-layer 24 --n-head 16 --n-embd 1024 \
  --micro-bs 8 --grad-accum 64 --total-tokens 15e9 --warmup 2000 --lr 6e-4 \
  --data-mix "${MIX}" --nope --scale-learnable --qk-norm --tag nope-mix "$@"
echo "ALL DONE"
