#!/bin/bash
#SBATCH --job-name=lm-v2-clamp
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-v2-clamp-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-v2-clamp-%j.err

# Inference-time scale-extrapolation test: identical training to the
# V2-IFT pair (12399063; clamp is inactive at train lengths), eval with
# the log-length input capped at the training block. If seed 0's far-OOD
# explosion (6.28 @8k) vanishes -> the LM instability is eval-time
# (log n)^gamma extrapolation, not training dynamics.

set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

for SEED in 0 1; do
  uv run --project softmax-is-meh/triton --no-sync python \
    softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
    --data stack --attn asflashn --as-v2 --ift-grad --clamp-logn \
    --q 4 --seed $SEED \
    --eval-blocks 512 1024 2048 4096 8192 16384 \
    || echo ">>> seed $SEED failed; continuing"
done

echo "ALL DONE"
