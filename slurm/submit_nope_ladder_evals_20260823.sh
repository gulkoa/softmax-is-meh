#!/bin/bash
# Auto-evals behind the NoPE ladder lanes (2026-08-23). Per lane:
#  - it2 gauntlet (policy: full probe battery after every patch):
#    multiturn + deep-context recall on -it-dr-it2
#  - code-bench (same protocol as the 08-11 zoo sweep, same sweep dir):
#    it2, code-r, code-GRPO final, reason-GRPO final — MBPP-100+HumanEval-100
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
STAGE=slurm/code_rl_stage_h100.sh
STJ=${FW}/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0
SDP=${FW}/ckpt_gpt2-sdpa-nope-nope-mix-lr0.0002_s0

evals () {  # evals <base> <it2-dep-or-empty> <lane-tail-dep>
  local B=$1 DI=$2 DT=$3 di=""
  [[ -n "$DI" ]] && di="-d afterok:${DI}"
  M=$(sbatch --parsable -t 01:00:00 $di ${STAGE} eval_multiturn_stilt.py ${B}-it-dr-it2.pt --system-native)
  D=$(sbatch --parsable -t 01:00:00 $di ${STAGE} eval_deepctx_recall_stilt.py ${B}-it-dr-it2.pt)
  Z=$(sbatch --parsable -d afterany:${DT} slurm/bench_code_zoo_h100.sh both \
      ${B}-it-dr-it2.pt ${B}-it-dr-code-r.pt ${B}-it-dr-code-r_grpo_final.pt ${B}-it-dr-code-r_reason_final.pt)
  echo "$(basename $B): it2-multiturn=${M} it2-deepctx=${D} codebench=${Z}"
}
evals ${STJ} "" 13816627   # stj it2 already COMPLETED (ckpt exists)
evals ${SDP} 13815759 13816632
