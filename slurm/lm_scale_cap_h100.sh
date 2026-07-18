#!/bin/bash
#SBATCH --job-name=lm-scale-cap
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-scale-cap-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-scale-cap-%j.err

# Equilibrium-selection fix test: soft-cap the AS scale multiplier
# (tanh at 15 — inside the healthy seed's realized band) during TRAINING
# so extreme-sharpness equilibria (bad seed: scale p99 ~58 via gamma->+2)
# cannot be selected. 4 seeds x V2-IFT with --scale-cap 15 --scale-diag.
# Success = no far-OOD explosions across seeds with ID loss preserved.

set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

for SEED in 0 1 2 3; do
  uv run --project softmax-is-meh/triton --no-sync python \
    softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
    --data stack --attn asflashn --as-v2 --ift-grad \
    --scale-cap 15 --scale-diag --q 4 --seed $SEED \
    --eval-blocks 512 1024 2048 4096 8192 16384 \
    || echo ">>> seed $SEED failed; continuing"
done

echo "ALL DONE"
