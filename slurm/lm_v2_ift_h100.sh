#!/bin/bash
#SBATCH --job-name=lm-v2-ift
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-v2-ift-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-v2-ift-%j.err

# 19M code-LM V2 seeds with the kernel's normalized-IFT backward
# (ift_grad=True, validated job 12396741). Tests whether the V2 far-OOD
# seed instability (seed 0 exploded to 7.5 nats @8k with the detached
# backward, job 12342643) is the same argmax-discontinuity mechanism as
# the MQMTAR collapse (finding 2026-07-16).

set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

for SEED in 0 1; do
  uv run --project softmax-is-meh/triton --no-sync python \
    softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
    --data stack --attn asflashn --as-v2 --ift-grad --q 4 --seed $SEED \
    --eval-blocks 512 1024 2048 4096 8192 16384 \
    || echo ">>> seed $SEED failed; continuing"
done

echo "ALL DONE"
