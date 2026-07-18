#!/bin/bash
#SBATCH --job-name=lm-v2-betacap
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-v2-betacap-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-v2-betacap-%j.err

# Sharpness-ceiling fix test: scale-diag showed the far-OOD-fragile seed
# places extreme sharpness early (L1 scale p99~60 from beta_max~2.1);
# beta-cap 1.0 bounds the ceiling DURING TRAINING (max scale ~
# 1 + (log n)^gamma). If seed 0 heals at 8k/16k while ID stays intact,
# bounded-beta completes the AS-for-LM recipe.

set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

for SEED in 0 1; do
  uv run --project softmax-is-meh/triton --no-sync python \
    softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
    --data stack --attn asflashn --as-v2 --ift-grad --beta-cap 1.0 \
    --scale-diag --q 4 --seed $SEED \
    --eval-blocks 512 1024 2048 4096 8192 16384 \
    || echo ">>> seed $SEED failed; continuing"
done

echo "ALL DONE"
