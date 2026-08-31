#!/bin/bash
#SBATCH --job-name=nope-qk-twin
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/nope-qk-twin-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/nope-qk-twin-%j.err
# 355M FAIR-TWIN RERUN (qknorm gate verdict promotion rule): identical
# to the original sdpa-nope arm (which diverged at 6e-4 and 3e-4;
# survived only at 2e-4 -> 12.82) EXCEPT +--qk-norm. Decides whether
# the 355M robustness asymmetry is attention-logit growth.
# Self-resuming; FINAL-guarded chunks no-op.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7"
MIX="${MIX},math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15"
MIX="${MIX},code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn sdpa --n-layer 24 --n-head 16 --n-embd 1024 \
  --micro-bs 8 --grad-accum 64 --total-tokens 15e9 --warmup 2000 --lr 6e-4 \
  --data-mix "${MIX}" --nope --scale-learnable --qk-norm --tag nope-mix "$@"
echo "ALL DONE"
