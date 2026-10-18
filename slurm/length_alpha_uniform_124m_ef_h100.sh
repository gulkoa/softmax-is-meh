#!/bin/bash
#SBATCH --job-name=len-a-uni-ef
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/length-alpha-uniform-124m-ef-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/length-alpha-uniform-124m-ef-%j.err
# Uniform 0.025 alpha grid, 0.05-0.50, for the 124M modern pair E (Stieltjes)
# and F (softmax), both NoPE + QK-norm, lr 1e-3 (record
# thesis/experiments/2026-10-17-length-alpha-uniform-grid.md). Versioned
# copy of p2_length_alpha_edge_h100.sh with E/F in place of the 355M pair.
# Its 0.25/0.3/0.35/0.5 columns must reproduce 14072702 (E) / 14072703 (F).
# Refuses to run unless both checkpoints are FINAL.
set -euo pipefail
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
E=$D/ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt
F=$D/ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s0.pt
/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python - "$E" "$F" <<'PY'
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
for C in "$E" "$F"; do
  echo "=== $(basename "$C")"
  uv run --project softmax-is-meh/triton --no-sync python \
    softmax-is-meh/triton/dev_scripts/eval_longctx_scaled.py "$C" \
    --alpha 0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5
done
echo "ALL DONE"
