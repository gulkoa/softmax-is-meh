#!/bin/bash
# ONE-SHOT: nope-it eval gauntlet, dependency-queued behind the SFT
# chain tail (13326428). Gates before any behavior patch / ship:
#   deep-context recall (the artifact's raison d'être) + positional
#   -it control, multiturn 8-case back-ref, humaneval code-regression.
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
TAIL=13326428
NOPEIT=${FW}/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0-it.pt
POSIT=${FW}/ckpt_gpt2-stj-q4-medium-mix_s0-it.pt

J1=$(sbatch --parsable -t 01:00:00 -d afterany:${TAIL} slurm/code_rl_stage_h100.sh \
  eval_deepctx_recall_stilt.py ${NOPEIT})
J2=$(sbatch --parsable -t 01:00:00 -d afterany:${TAIL} slurm/code_rl_stage_h100.sh \
  eval_deepctx_recall_stilt.py ${POSIT})
J3=$(sbatch --parsable -t 01:00:00 -d afterany:${TAIL} slurm/code_rl_stage_h100.sh \
  eval_multiturn_stilt.py ${NOPEIT} --system-native)
J4=$(sbatch --parsable -t 01:00:00 -d afterany:${TAIL} slurm/code_rl_stage_h100.sh \
  eval_code_stilt.py ${NOPEIT} --dataset humaneval --limit 100)
echo "gauntlet: deepctx=${J1} deepctx-ctrl=${J2} multiturn=${J3} humaneval=${J4}"
