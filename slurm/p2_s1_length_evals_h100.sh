#!/bin/bash
#SBATCH --job-name=p2s1-len-ev
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/p2s1-length-evals-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/p2s1-length-evals-%j.err
# Pre-registered length eval of the 355M seed-1 matched pair (NoPE + QK-norm,
# lr 6e-4; record thesis/experiments/2026-10-14-p2-seed1-matched-pair.md).
# Same protocol as 09-15 (eval_longctx_scaled.py, unchanged since 09-14;
# alpha 0.25,0.3,0.35,0.5,1.0) on both seed-1 arms, then both seed-0 arms
# re-run in the same job as a consistency check against 09-15
# (best-alpha 8x: Stieltjes 152.7, softmax 169.9).
# Refuses to run unless BOTH seed-1 checkpoints are FINAL (chains end afterany).
set -euo pipefail
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
S1=$D/ckpt_gpt2-stj-q4-nope-qk-nope-mix-lr0.0006_s1.pt
T1=$D/ckpt_gpt2-sdpa-nope-qk-nope-mix-lr0.0006_s1.pt
S0=$D/ckpt_gpt2-stj-q4-nope-qk-nope-mix-lr0.0006_s0.pt
T0=$D/ckpt_gpt2-sdpa-nope-qk-nope-mix-lr0.0006_s0.pt
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
for C in "$S1" "$T1" "$S0" "$T0"; do
  echo "=== $(basename "$C")"
  uv run --project softmax-is-meh/triton --no-sync python \
    softmax-is-meh/triton/dev_scripts/eval_longctx_scaled.py "$C" \
    --alpha 0.25,0.3,0.35,0.5,1.0
done
echo "ALL DONE"
