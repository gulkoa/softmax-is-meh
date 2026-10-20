#!/bin/bash
#SBATCH --job-name=stilt12
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12-%j.err
# stilt.1.2 355M pair, launch copy of stilt12_pair_h100.sh after the 10-20
# amendment (thesis/findings/2026-10-14-plan-stilt12-355m-pair.md, "Amendment
# (2026-10-20, before launch)"). Changes from v1, nothing else:
#   - data: 100 % FineWeb sample-100BT in stilt-tok-32k-v1 shards (last shard =
#     validation; the old FineWeb-Edu validation documents removed);
#   - budget fixed at 14,020,853,435 tokens = 26,742 steps (v1's helper needs
#     gpt2_total_tokens, which the new meta.json does not have, and would fall
#     back to 1.4e10 without a warning);
#   - 4 h chunks (submit with --time-min, see submit_stilt12_pair_v2.sh) and
#     --ckpt-every 250 (the 10-16 queue lesson).
# PREPARED 2026-10-20, NOT SUBMITTED: the GPU spend is user-gated.
# Usage: sbatch stilt12_pair_v2_h100.sh <stj|sdpa> [extra trainer args]
set -euo pipefail
ARM=${1:?stj|sdpa}; shift
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline HF_HUB_OFFLINE=1
MIX="web=/fs/scratch/PAS2836/alexg/fineweb_100bt_tok32k:1.0"
TOK=14020853435
echo "data: ${MIX}; token budget: ${TOK}"
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "$ARM" --n-layer 24 --n-head 16 --n-embd 1024 \
  --micro-bs 8 --grad-accum 64 --total-tokens "$TOK" --warmup 2000 --lr 1e-3 \
  --data-mix "${MIX}" --nope --qk-norm --scale-cap 0 --scale-learnable \
  --modern --compile --ckpt-every 250 --tag stilt12 "$@"
echo "ALL DONE"
