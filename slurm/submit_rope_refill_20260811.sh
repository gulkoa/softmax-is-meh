#!/bin/bash
# RoPE 355M chain refill (2026-08-11). The original 18-chunk ladder from
# submit_postnope_pipeline.sh exhausted at step 18,880/28,610 on 08-09
# (~880 steps per 2h chunk). Remaining ~9,730 steps => 12 chunks with
# self-guarded no-op drain once the trainer hits max steps.
# Also resubmits the #37 sdpa-side scaled-length eval against the
# CORRECT final ckpt (lr0.0002 restart; the pipeline's E2 pointed at the
# stale lr0.0003 name and produced nothing).
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh

FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
ALPHAS="0.25,0.3,0.35,0.5,1.0"

E2=$(sbatch --parsable -t 01:00:00 slurm/code_rl_stage_h100.sh \
  eval_longctx_scaled.py ${FW}/ckpt_gpt2-sdpa-nope-nope-mix-lr0.0002_s0.pt \
  --alpha ${ALPHAS})

PREV=$(sbatch --parsable slurm/gpt2_rope_h100.sh stj)
CHAIN="${PREV}"
for i in $(seq 2 12); do
  PREV=$(sbatch --parsable -d afterany:${PREV} slurm/gpt2_rope_h100.sh stj)
  CHAIN="${CHAIN} ${PREV}"
done

echo "E2-redo ${E2}"
echo "rope-refill ${CHAIN}"
