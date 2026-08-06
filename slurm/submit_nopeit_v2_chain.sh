#!/bin/bash
# ONE-SHOT: nope-it v2 — v1 recipe (full smoltalk, untrimmed, ctx 2048)
# + 8k synthetic deep-recall convos (explicit demand for deep
# back-reference; v1 negative in finding 2026-08-06-nopeit-v1). The
# --identity hook is used as a generic extra-convos channel, upsample 1.
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
A="sft_multiturn_stilt.py ${FW}/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0.pt \
   --ctx 2048 --dataset full --min-assistant-turns 1 --max-examples 300000 \
   --tokens 5e8 --out-suffix=-it-dr \
   --identity /users/PAS2402/alexg/softmax/softmax-is-meh/results/deeprecall_convos.json \
   --identity-upsample 1"
J1=$(sbatch --parsable -t 04:00:00 slurm/code_rl_stage_h100.sh $A)
J2=$(sbatch --parsable -t 04:00:00 -d afterany:$J1 slurm/code_rl_stage_h100.sh $A)
J3=$(sbatch --parsable -t 04:00:00 -d afterany:$J2 slurm/code_rl_stage_h100.sh $A)
echo "v2 chain: $J1 -> $J2 -> $J3"

NOPEIT=${FW}/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0-it-dr.pt
G1=$(sbatch --parsable -t 01:00:00 -d afterany:$J3 slurm/code_rl_stage_h100.sh \
  eval_deepctx_recall_stilt.py ${NOPEIT})
G2=$(sbatch --parsable -t 01:00:00 -d afterany:$J3 slurm/code_rl_stage_h100.sh \
  eval_multiturn_stilt.py ${NOPEIT} --system-native)
G3=$(sbatch --parsable -t 01:00:00 -d afterany:$J3 slurm/code_rl_stage_h100.sh \
  eval_code_stilt.py ${NOPEIT} --dataset humaneval --limit 100)
echo "v2 gauntlet: deepctx=$G1 multiturn=$G2 humaneval=$G3"
