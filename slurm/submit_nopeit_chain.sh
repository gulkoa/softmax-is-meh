#!/bin/bash
# ONE-SHOT: nope-it untrimmed full-smoltalk SFT as a 3x4h resumable
# chain. Bash (not zsh): unquoted $A word-splits correctly here — the
# 2026-08-05 instant-fail was zsh passing the arg string as one word.
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
A="sft_multiturn_stilt.py ${FW}/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0.pt \
   --ctx 4096 --dataset full --min-assistant-turns 1 --max-examples 300000 \
   --tokens 5e8 --out-suffix=-it"
J1=$(sbatch --parsable -t 04:00:00 slurm/code_rl_stage_h100.sh $A)
J2=$(sbatch --parsable -t 04:00:00 -d afterany:$J1 slurm/code_rl_stage_h100.sh $A)
J3=$(sbatch --parsable -t 04:00:00 -d afterany:$J2 slurm/code_rl_stage_h100.sh $A)
echo "SFT chain: $J1 -> $J2 -> $J3"
