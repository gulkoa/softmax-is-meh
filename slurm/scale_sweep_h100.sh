#!/bin/bash
#SBATCH --job-name=scale-sweep
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/scale-sweep-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/scale-sweep-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

echo "########## PART 1: max-retrieval scale sweep (d_emb 256, 512) ##########"
SWEEP_DEMBS="256,512" \
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/maxretr_scale_sweep.py \
  || echo ">>> scale sweep reported failures; continuing"

echo ""
echo "########## PART 2: code-LM at ~3x params (12L 12H d768, stack) ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
  --data stack --attn sdpa --n-layer 12 --n-head 12 --n-embd 768 --iters 3000 \
  || echo ">>> big sdpa arm failed; continuing"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
  --data stack --attn flashn --q 4 --n-layer 12 --n-head 12 --n-embd 768 --iters 3000 \
  || echo ">>> big flashn arm failed; continuing"

echo "ALL DONE"
