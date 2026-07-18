#!/bin/bash
#SBATCH --job-name=lm-scale-diag
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-scale-diag-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-scale-diag-%j.err

# Content-driven scale-runaway probe: rerun the divergent V2-IFT seed
# pair with realized per-layer scale statistics printed at every eval
# context. Prediction if the hypothesis holds: seed 0's realized scale
# (p99/max) blows up at ctx >= 4096 in the layers driving the loss
# explosion, seed 1's stays bounded.

set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

for SEED in 0 1; do
  uv run --project softmax-is-meh/triton --no-sync python \
    softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
    --data stack --attn asflashn --as-v2 --ift-grad --scale-diag \
    --q 4 --seed $SEED \
    --eval-blocks 512 1024 2048 4096 8192 16384 \
    || echo ">>> seed $SEED failed; continuing"
done

echo "ALL DONE"
