#!/bin/bash
#SBATCH --job-name=len-a-uni-355
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/length-alpha-uniform-355m-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/length-alpha-uniform-355m-%j.err
# Uniform 0.025 alpha grid for the 355M length comparisons (record
# thesis/experiments/2026-10-17-length-alpha-uniform-grid.md). Versioned
# copy of p2_length_alpha_edge_h100.sh: adds the Stieltjes NoPE base (HF
# round-trip copy; its scratch original is gone) and only the alpha points
# the earlier jobs lack, plus 0.35 as the reproduction column (must match
# 15534295 and 13297095). Scored on the union with 15534295, 15637051 and
# 13297095: 0.15-0.50 every 0.025 for the four matched-pair checkpoints.
# Refuses to run unless BOTH seed-1 checkpoints are FINAL (kept from the copy).
set -euo pipefail
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
S1=$D/ckpt_gpt2-stj-q4-nope-qk-nope-mix-lr0.0006_s1.pt
T1=$D/ckpt_gpt2-sdpa-nope-qk-nope-mix-lr0.0006_s1.pt
S0=$D/ckpt_gpt2-stj-q4-nope-qk-nope-mix-lr0.0006_s0.pt
T0=$D/ckpt_gpt2-sdpa-nope-qk-nope-mix-lr0.0006_s0.pt
N0=/fs/scratch/PAS2836/alexg/hf_probe_355m_nope/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0.fromhf.pt
/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python - "$S1" "$T1" <<'PY'
import sys, torch
ok = True
for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    a = ck["args"]
    last = int(a["total_tokens"] // (a["micro_bs"] * a["grad_accum"] * a["ctx"])) - 1
    print(p.split("/")[-1], "step", ck["step"], "final step", last, flush=True)
    ok = ok and ck["step"] >= last
sys.exit(0 if ok else 1)
PY
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
for C in "$S1" "$T1" "$S0" "$T0" "$N0"; do
  echo "=== $(basename "$C")"
  uv run --project softmax-is-meh/triton --no-sync python \
    softmax-is-meh/triton/dev_scripts/eval_longctx_scaled.py "$C" \
    --alpha 0.175,0.225,0.275,0.325,0.35,0.375,0.4,0.425,0.45,0.475
done
echo "ALL DONE"
