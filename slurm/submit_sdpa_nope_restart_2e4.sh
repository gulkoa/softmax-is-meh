#!/bin/bash
# ONE-SHOT: cut over the failing sdpa-nope arm (recurrent mega-spikes at
# 3e-4; tripwire in experiments/2026-07-25-nope-355m-pair.md) to a fresh
# 2e-4 chain. Run ONCE, only after the tripwire fires.
#
# Effects (all pre-registered): cancels the sdpa longctx eval (only
# pipeline job that needs sdpa-nope) and the old 3e-4 chain — the
# post-nope pipeline then fires early, which is benign (stj evals valid;
# round-3/P1/RoPE are sdpa-nope-independent). Submits a 20x2h 2e-4
# chain (new label/ckpt via lr-in-label) + the sdpa longctx eval behind
# its tail.

set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh

OLD_EVAL=13297096
OLD_CHAIN="13246256 13246257 13246258 13246259 13246260 13246261 13246262 13246263 13246264 13246265 13246266 13246267 13246268 13246269 13246270 13246271 13246272"
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt

echo "== cancel sdpa longctx eval + old 3e-4 chain =="
scancel ${OLD_EVAL}
scancel ${OLD_CHAIN}

echo "== submit 2e-4 chain (20 x 2h, self-resuming/guarded) =="
PREV=$(sbatch --parsable slurm/gpt2_medium2_h100.sh sdpa \
  --tag nope-mix --nope --scale-learnable --lr 2e-4)
CHAIN="${PREV}"
for i in $(seq 2 20); do
  PREV=$(sbatch --parsable -d afterany:${PREV} slurm/gpt2_medium2_h100.sh sdpa \
    --tag nope-mix --nope --scale-learnable --lr 2e-4)
  CHAIN="${CHAIN} ${PREV}"
done

echo "== sdpa longctx eval behind new tail =="
EV=$(sbatch --parsable -t 01:00:00 -d afterany:${PREV} slurm/code_rl_stage_h100.sh \
  eval_longctx_scaled.py ${FW}/ckpt_gpt2-sdpa-nope-nope-mix-lr0.0002_s0.pt \
  --alpha 0.25,0.3,0.35,0.5,1.0)

echo ""
echo "NEW CHAIN: ${CHAIN}"
echo "NEW TAIL:  ${PREV}"
echo "SDPA LONGCTX EVAL: ${EV}"
