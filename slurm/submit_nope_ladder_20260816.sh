#!/bin/bash
# Full post-training ladder on the NoPE 355M pair (user directive
# 2026-08-16: "add all other steps to nope model as well - rl, sft, etc").
# Treatment parity (standing policy): every stage applied IDENTICALLY to
# both arms — same scripts, data, budgets, seed 0 — queued together as
# two dependency lanes. Mirrors the positional stilt.1 line:
#   -it-dr  (SFT v2: full smoltalk untrimmed @2048 + deep-recall slice)
#   -it-dr-it2     behavior-patch SFT (system-native template + identity v3)
#   -it-dr-code-r  code-SFT in GRPO prompt format, reasoning targets
#   probe -> code-GRPO (400 steps, micro-bs 8) on the solvable set
#   reason probe -> reasoning-GRPO (fused verify+judge reward)
# stj lane starts at it2 (its -it-dr exists); sdpa lane first trains its
# own -it-dr (3 x 4h self-resuming chunks) so the twin is a true parity copy.
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
RES=/users/PAS2402/alexg/softmax/softmax-is-meh/results
STJ=${FW}/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0
SDP=${FW}/ckpt_gpt2-sdpa-nope-nope-mix-lr0.0002_s0
STAGE=slurm/code_rl_stage_h100.sh

lane () {  # lane <base-label> <dep-job-or-empty>
  local B=$1 DEP=$2 d=""
  [[ -n "$DEP" ]] && d="-d afterany:${DEP}"
  local IT=${B}-it-dr.pt CR=${B}-it-dr-code-r.pt
  I2=$(sbatch --parsable -t 02:00:00 $d ${STAGE} sft_stilt_it2.py ${IT} \
        --identity ${RES}/identity_sft_v3.json)
  CS=$(sbatch --parsable -t 01:30:00 $d ${STAGE} sft_code_stilt.py ${IT} \
        --reasoning --synthetic ${RES}/synth_code_tasks_v1.json)
  PR=$(sbatch --parsable -t 02:00:00 -d afterany:${CS} slurm/grpo_code2_h100.sh ${CR} --probe)
  RL=$(sbatch --parsable -d afterany:${PR} slurm/grpo_code2_h100.sh ${CR} \
        --solvable ${B}-it-dr-code-r_mbpp_probe.json \
        --synthetic ${RES}/synth_code_tasks_v1.json --steps 400 --micro-bs 8)
  RP=$(sbatch --parsable -t 03:00:00 -d afterany:${RL} ${STAGE} grpo_reason_stilt.py ${CR} \
        --logic ${RES}/logic_tasks_v1.json --probe)
  RR=$(sbatch --parsable -t 08:00:00 -d afterany:${RP} ${STAGE} grpo_reason_stilt.py ${CR} \
        --logic ${RES}/logic_tasks_v1.json \
        --solvable ${B}-it-dr-code-r_reason_probe.json --steps 400 --micro-bs 8)
  echo "$(basename $B): it2=${I2} code-r=${CS} probe=${PR} codeRL=${RL} rprobe=${RP} reasonRL=${RR}"
}

# --- sdpa lane: SFT v2 first (identical to the stj v2 chain of 08-06) ---
A="sft_multiturn_stilt.py ${SDP}.pt --ctx 2048 --dataset full --min-assistant-turns 1 \
   --max-examples 300000 --tokens 5e8 --out-suffix=-it-dr \
   --identity ${RES}/deeprecall_convos.json --identity-upsample 1"
J1=$(sbatch --parsable -t 04:00:00 ${STAGE} $A)
J2=$(sbatch --parsable -t 04:00:00 -d afterany:$J1 ${STAGE} $A)
J3=$(sbatch --parsable -t 04:00:00 -d afterany:$J2 ${STAGE} $A)
echo "sdpa-nope SFT v2 chain: $J1 -> $J2 -> $J3"
# twin gauntlet (parity with the stj v2 gauntlet)
SDPIT=${SDP}-it-dr.pt
G1=$(sbatch --parsable -t 01:00:00 -d afterany:$J3 ${STAGE} eval_deepctx_recall_stilt.py ${SDPIT})
G2=$(sbatch --parsable -t 01:00:00 -d afterany:$J3 ${STAGE} eval_multiturn_stilt.py ${SDPIT} --system-native)
echo "sdpa-nope gauntlet: deepctx=$G1 multiturn=$G2"

lane ${STJ} ""
lane ${SDP} ${J3}
