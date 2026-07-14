#!/bin/bash
#SBATCH --job-name=as-v2-targeted
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/as-v2-targeted-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/as-v2-targeted-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

echo "########## PART 1: V2 (per-pos tanh gamma) at the FAILING d512 cell ##########"
AS_V2=1 SWEEP_DEMBS="512" SWEEP_QORDERS="4" SWEEP_LRS="3e-4,1e-3,3e-3" \
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/asstj_16k_stretch.py \
  || echo ">>> part 1 failures; continuing"

echo ""
echo "########## PART 2: V2 sanity at d256 (must preserve the win) ##########"
AS_V2=1 SWEEP_DEMBS="256" SWEEP_QORDERS="4" SWEEP_LRS="3e-4" \
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/asstj_16k_stretch.py \
  || echo ">>> part 2 failures; continuing"

echo ""
echo "########## PART 3: code-LM V2 arms (19M x2 seeds — variance check) ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
  --data stack --attn asflashn --as-v2 --q 4 --seed 0 \
  --eval-blocks 512 1024 2048 4096 8192 16384 \
  || echo ">>> 19M v2 s0 failed; continuing"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
  --data stack --attn asflashn --as-v2 --q 4 --seed 1 \
  --eval-blocks 512 1024 2048 4096 8192 16384 \
  || echo ">>> 19M v2 s1 failed; continuing"

echo "ALL DONE"
