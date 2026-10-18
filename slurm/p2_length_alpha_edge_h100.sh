#!/bin/bash
#SBATCH --job-name=p2-len-edge
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/p2-length-alpha-edge-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/p2-length-alpha-edge-%j.err
# Post-hoc alpha sweep below 0.25 for the 355M matched pair at both seeds
# (record thesis/experiments/2026-10-17-p2-length-alpha-edge.md). In the
# pre-registered evals (15534295; 09-15: 14227114, 14559792) softmax's best
# alpha is the lowest swept value, 0.25, at every length. A versioned copy
# of p2_s1_length_evals_h100.sh: same checkpoints, order and eval script
# (unchanged since 09-14); only the alpha grid differs. Its 0.25 column must
# reproduce 15534295. alpha = 0 is the 'off' column.
# Refuses to run unless BOTH seed-1 checkpoints are FINAL (kept from the copy).
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
    --alpha 0.05,0.1,0.15,0.2,0.25
done
echo "ALL DONE"
