#!/bin/bash
#SBATCH --job-name=stj-alibi-ev
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stj-alibi-evals-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stj-alibi-evals-%j.err
# Pre-registered evals for Stieltjes + ALiBi vs softmax + ALiBi (stjalibi worktree).
# v2: refuses to run unless the checkpoint is FINAL.
set -euo pipefail
# v2 guard: evaluate only a FINAL checkpoint (the training chain ends afterany)
/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python - <<'PY'
import sys, torch
ck = torch.load("/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-alibi-qk-modern-nocap-qkpilot-lr0.001_s0.pt",
                map_location="cpu", weights_only=False, mmap=True)
a = ck["args"]; last = int(a["total_tokens"] // (a["micro_bs"] * a["grad_accum"] * 1024)) - 1
print("ckpt step", ck["step"], "final step", last, flush=True)
sys.exit(0 if ck["step"] >= last else 1)
PY
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
S=$D/ckpt_gpt2-stj-q4-alibi-qk-modern-nocap-qkpilot-lr0.001_s0.pt
A=$D/ckpt_gpt2-sdpa-alibi-qk-modern-nocap-qkpilot-lr0.001_s0.pt
PY="uv run --project softmax-is-meh/triton --no-sync python /users/PAS2402/alexg/softmax/.worktrees/stjalibi/triton/dev_scripts"
echo "=== (2) PG-19 ppl + BPB, identical text (Stieltjes + ALiBi, 8 lambda iterations)"
$PY/eval_longctx_scaled.py $S --text-chars 2400000 --alpha 0.25,0.3,0.35,0.5 --stj-iters 8
for T in noquestion standard; do
  echo "=== (3) passkey, template $T: Stieltjes + ALiBi"; HF_HUB_OFFLINE=1 $PY/eval_passkey.py $S --template $T --alpha 0,0.35 --trials 20 --stj-iters 8 --show-fails 2
done
echo "=== (3) passkey, template noquestion: softmax + ALiBi (re-run for the primary comparison)"
HF_HUB_OFFLINE=1 $PY/eval_passkey.py $A --template noquestion --alpha 0,0.35 --trials 20 --show-fails 2
echo "ALL DONE"
