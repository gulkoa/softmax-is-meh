#!/bin/bash
#SBATCH --job-name=heff-confirm
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/heff-confirm-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/heff-confirm-%j.err

# H_eff stride-bug reattribution confirmation (task #39; finding
# 2026-08-04-triton-heff-stride-bug-audit.md). Same ckpt/N/num_iter,
# Triton backend, batch-size 1 then 2:
#   B=1 ≈ ref (0.976)  => April "intermediate-N dip" = the stride bug.
#   B=2 low            => answers the magnitude question.
# --num-iter-override is REQUIRED: eval_accuracy silently couples
# num_iter to backend (10 triton / 3 ref) otherwise.

set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
export WANDB_MODE=offline

for BS in 1 2; do
  echo "=== batch-size ${BS} ==="
  uv run --project triton --no-sync python nanogpt/eval_accuracy.py \
    --checkpoint results/subtle_needle_1layer_stieltjes_q4.0_seq128_fixedcap_ascend/model.pt \
    --task needle --needle-margin subtle --attn stieltjes --q 4.0 \
    --seq-len 2048 --max-arr-len 120 --val-samples 1000 \
    --use-triton --num-iter-override 3 --batch-size "${BS}"
done

echo "ALL DONE"
