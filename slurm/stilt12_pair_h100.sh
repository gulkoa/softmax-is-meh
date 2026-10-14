#!/bin/bash
#SBATCH --job-name=stilt12
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12-%j.err
# stilt.1.2 355M pair (plan thesis/findings/2026-10-14-plan-stilt12-355m-pair.md):
# the 124M E/F recipe at 355M on stilt-tok-32k-v1 — NoPE, QK-norm, modern
# backbone (RMSNorm/SwiGLU/no bias, tied head), compiled MLP/norm paths, no
# scale cap, lr 1e-3, warmup 2000, 524,288 tok/step, budget = the same bytes as
# 15B GPT-2 tokens (computed from the shard meta.json). Self-resuming 16 h
# chunks, FINAL-guarded. Runs the tok32k worktree trainer.
# PREPARED 2026-10-14, NOT SUBMITTED — user-gated (data-mix decision).
# Usage: sbatch stilt12_pair_h100.sh <stj|sdpa> [extra trainer args]
set -euo pipefail
ARM=${1:?stj|sdpa}; shift
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline HF_HUB_OFFLINE=1
B=/fs/scratch/PAS2836/alexg
MIX=${STILT12_MIX:-"web=${B}/fineweb_edu_10bt_tok32k:0.7,math=${B}/finemath_4plus_tok32k:0.15,code=${B}/codeparrot_py_tok32k:0.15"}
TOK=$(python3 - "$MIX" <<'PY' || echo 1.4e10
import json, sys
r = 0.0
for part in sys.argv[1].split(","):
    _, rest = part.split("=", 1); d, w = rest.rsplit(":", 1)
    m = json.load(open(f"{d}/meta.json")); r += float(w) * m["total_tokens"] / m["gpt2_total_tokens"]
print(f"{15e9 * r:.0f}")
PY
)
echo "token budget (same bytes as 15e9 GPT-2 tokens): $TOK"
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/train_gpt2_stieltjes.py \
  --attn "$ARM" --n-layer 24 --n-head 16 --n-embd 1024 \
  --micro-bs 8 --grad-accum 64 --total-tokens "$TOK" --warmup 2000 --lr 1e-3 \
  --data-mix "${MIX}" --nope --qk-norm --scale-cap 0 --scale-learnable \
  --modern --compile --tag stilt12 "$@"
echo "ALL DONE"
