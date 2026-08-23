#!/bin/bash
# Fix for submit_nope_ladder_20260816.sh: the reasoning-format code-SFT
# takes --data results/reasoning_code_sft_v1.json (rows carry `think`),
# not --synthetic (KeyError 'think', job 13815754). Resubmits code-r and
# everything downstream for both lanes; it2 jobs (13815753/59) untouched.
# sdpa lane still waits on its SFT v2 chain tail (13815750).
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
RES=/users/PAS2402/alexg/softmax/softmax-is-meh/results
STAGE=slurm/code_rl_stage_h100.sh

lane () {  # lane <base-label> <dep-job-or-empty>
  local B=$1 DEP=$2 d=""
  [[ -n "$DEP" ]] && d="-d afterany:${DEP}"
  local IT=${B}-it-dr.pt CR=${B}-it-dr-code-r.pt
  CS=$(sbatch --parsable -t 01:30:00 $d ${STAGE} sft_code_stilt.py ${IT} \
        --reasoning --data ${RES}/reasoning_code_sft_v1.json)
  PR=$(sbatch --parsable -t 02:00:00 -d afterok:${CS} slurm/grpo_code2_h100.sh ${CR} --probe)
  RL=$(sbatch --parsable -d afterok:${PR} slurm/grpo_code2_h100.sh ${CR} \
        --solvable ${B}-it-dr-code-r_mbpp_probe.json \
        --synthetic ${RES}/synth_code_tasks_v1.json --steps 400 --micro-bs 8)
  RP=$(sbatch --parsable -t 03:00:00 -d afterany:${RL} ${STAGE} grpo_reason_stilt.py ${CR} \
        --logic ${RES}/logic_tasks_v1.json --probe)
  RR=$(sbatch --parsable -t 08:00:00 -d afterok:${RP} ${STAGE} grpo_reason_stilt.py ${CR} \
        --logic ${RES}/logic_tasks_v1.json \
        --solvable ${B}-it-dr-code-r_reason_probe.json --steps 400 --micro-bs 8)
  echo "$(basename $B): code-r=${CS} probe=${PR} codeRL=${RL} rprobe=${RP} reasonRL=${RR}"
}
lane ${FW}/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0 ""
lane ${FW}/ckpt_gpt2-sdpa-nope-nope-mix-lr0.0002_s0 13815750
