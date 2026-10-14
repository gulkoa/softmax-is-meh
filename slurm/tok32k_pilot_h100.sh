#!/bin/bash
#SBATCH --job-name=tok32k-pilot
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/tok32k-pilot-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/tok32k-pilot-%j.err
# Tokenizer pilot (plan 2026-10-14-plan-tokenizer-32k): backbone E/F recipe
# (124M, NoPE, qk-norm, modern, lr 1e-3) on the stilt-tok-32k-v1 shards, token
# budget = the SAME BYTES as E/F's 5B GPT-2 tokens (computed from meta.json).
# Usage: sbatch tok32k_pilot_h100.sh <stj|sdpa> [extra trainer args]
set -euo pipefail
ARM=${1:?stj|sdpa}; shift
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline HF_HUB_OFFLINE=1
B=/fs/scratch/PAS2836/alexg
MIX="web=${B}/fineweb_edu_10bt_tok32k:0.7,math=${B}/finemath_4plus_tok32k:0.15,code=${B}/codeparrot_py_tok32k:0.15"
TOK=$(python3 - <<'PY' || echo 5e9
import json
B = "/fs/scratch/PAS2836/alexg"; w = {"fineweb_edu_10bt": 0.7, "finemath_4plus": 0.15, "codeparrot_py": 0.15}
r = 0.0
for d, v in w.items():
    m = json.load(open(f"{B}/{d}_tok32k/meta.json")); r += v * m["total_tokens"] / m["gpt2_total_tokens"]
print(f"{5e9 * r:.0f}")
PY
)
echo "token budget (same bytes as 5e9 GPT-2 tokens): $TOK"
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "$ARM" --n-layer 12 --n-head 12 --n-embd 768 \
  --micro-bs 16 --grad-accum 32 --total-tokens "$TOK" --warmup 2000 --lr 1e-3 \
  --data-mix "${MIX}" --nope --qk-norm --scale-cap 0 --scale-learnable \
  --modern --tag qkpilot "$@"
echo "ALL DONE"
