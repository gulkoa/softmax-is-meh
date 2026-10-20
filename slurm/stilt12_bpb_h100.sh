#!/bin/bash
#SBATCH --job-name=stilt12-bpb
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12-bpb-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12-bpb-%j.err
# stilt.1.2 reading 1, bits per byte (thesis/findings/2026-10-14-plan-stilt12-355m-pair.md,
# amendment items 3 and 4; scored by scripts/score_stilt12.py --bpb <this log>):
#   - the stilt.1.2 pair on the first 1,500 documents of the new FineWeb
#     validation shard (fineweb_100bt_tok32k, last shard);
#   - the stilt.1.2 pair and the stilt.1.1 NoPE + QK-norm pair on the first
#     1,500 documents of the old FineWeb-Edu validation shard (shard_0039 of
#     each tokenizer's directory: the same documents).
# Every block is headed "=== <ckpt> | <mix>" for the scorer. Refuses to run
# unless both stilt.1.2 checkpoints are FINAL.
# PREPARED 2026-10-20, NOT SUBMITTED (part of the user-gated stilt.1.2 work).
set -euo pipefail
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
S12=ckpt_gpt2-stj-q4-nope-qk-modern-nocap-stilt12-lr0.001-stilt-tok-32k-v1_s0.pt
T12=ckpt_gpt2-sdpa-nope-qk-modern-nocap-stilt12-lr0.001-stilt-tok-32k-v1_s0.pt
S11=ckpt_gpt2-stj-q4-nope-qk-nope-mix-lr0.0006_s0.pt
T11=ckpt_gpt2-sdpa-nope-qk-nope-mix-lr0.0006_s0.pt
NEW="web=/fs/scratch/PAS2836/alexg/fineweb_100bt_tok32k:1.0"
OLD32="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt_tok32k:1.0"
OLD2="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:1.0"
/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python - "$D/$S12" "$D/$T12" <<'PY'
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
export WANDB_MODE=offline HF_HUB_OFFLINE=1
PY="uv run --project softmax-is-meh/triton --no-sync python /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/eval_bpb.py"
for c in $S12 $T12; do echo "=== $c | $NEW"; $PY "$D/$c" --mix "$NEW" --docs 1500; done
for c in $S12 $T12; do echo "=== $c | $OLD32"; $PY "$D/$c" --mix "$OLD32" --docs 1500; done
for c in $S11 $T11; do echo "=== $c | $OLD2"; $PY "$D/$c" --mix "$OLD2" --docs 1500; done
echo "ALL DONE"
