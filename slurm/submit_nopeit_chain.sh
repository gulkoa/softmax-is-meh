#!/bin/bash
# ONE-SHOT: nope-it untrimmed full-smoltalk SFT as a 3x4h resumable
# chain. Bash (not zsh): unquoted $A word-splits correctly here — the
# 2026-08-05 instant-fail was zsh passing the arg string as one word.
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
# ctx 2048, not 4096: training gradients through the stj kernel at
# N~4096 hit deterministic NaN at step ~250 (2026-08-05, job 13326305 —
# healthy 1.78->1.09 through 200 first; backward beyond the validated
# length band; kernel item in task #39). 2048 = the demonstrated-stable
# extrapolation band, still untrimmed for ~95% of full smoltalk.
A="sft_multiturn_stilt.py ${FW}/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0.pt \
   --ctx 2048 --dataset full --min-assistant-turns 1 --max-examples 300000 \
   --tokens 5e8 --out-suffix=-it"
J1=$(sbatch --parsable -t 04:00:00 slurm/code_rl_stage_h100.sh $A)
J2=$(sbatch --parsable -t 04:00:00 -d afterany:$J1 slurm/code_rl_stage_h100.sh $A)
J3=$(sbatch --parsable -t 04:00:00 -d afterany:$J2 slurm/code_rl_stage_h100.sh $A)
echo "SFT chain: $J1 -> $J2 -> $J3"
