#!/bin/bash
# Code-benchmark sweep over the full 355M family (2026-08-11).
# Tier A: finals/artifacts get MBPP-100 + HumanEval-100 (n=20, pass@1/10).
# Tier B: RL trajectory step-ckpts get HumanEval-100 only.
# 7 jobs, no dependencies (backfill around the RoPE chain).
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
STJ=${FW}/ckpt_gpt2-stj-q4-medium-mix_s0
SDP=${FW}/ckpt_gpt2-sdpa-medium-mix_s0
NOPE=${FW}/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0
R3=${FW}/ckpt_gpt2-stj-r3judge_s0

# --- Tier A: full battery ---
A1=$(sbatch --parsable slurm/bench_code_zoo_h100.sh both \
  ${STJ}-it.pt ${STJ}-it-it2.pt ${STJ}-it-mt.pt \
  ${STJ}-it-code.pt ${STJ}-it-code-r.pt ${STJ}-it-code-r_grpo300.pt)
A2=$(sbatch --parsable slurm/bench_code_zoo_h100.sh both \
  ${STJ}-it-code-r_grpo_final.pt ${STJ}-it-code-r_reason300.pt \
  ${STJ}-it-code-r_reason_final.pt ${R3}_grpo300.pt ${R3}_grpo_final.pt \
  ${NOPE}-it.pt)
A3=$(sbatch --parsable slurm/bench_code_zoo_h100.sh both \
  ${NOPE}-it-dr.pt ${SDP}-it.pt ${SDP}-it-it2.pt \
  ${SDP}-it-code.pt ${SDP}-it-code-r.pt ${SDP}-it-code-r_grpo_final.pt)

# --- Tier B: trajectory resolution, HumanEval only ---
B1=$(sbatch --parsable slurm/bench_code_zoo_h100.sh humaneval \
  ${STJ}-it-code-r_grpo50.pt ${STJ}-it-code-r_grpo100.pt \
  ${STJ}-it-code-r_grpo150.pt ${STJ}-it-code-r_grpo200.pt \
  ${STJ}-it-code-r_grpo250.pt ${STJ}-it-code-r_grpo350.pt \
  ${STJ}-it-code-r_reason50.pt ${STJ}-it-code-r_reason100.pt \
  ${STJ}-it-code-r_reason150.pt ${STJ}-it-code-r_reason200.pt)
B2=$(sbatch --parsable slurm/bench_code_zoo_h100.sh humaneval \
  ${STJ}-it-code-r_reason250.pt ${STJ}-it-code-r_reason350.pt \
  ${STJ}-it-code-r-s1rl_grpo50.pt ${STJ}-it-code-r-s1rl_grpo100.pt \
  ${STJ}-it-code-r-s1rl_grpo150.pt ${STJ}-it-code-r-s1rl_grpo200.pt \
  ${STJ}-it-code-r-s1rl_grpo250.pt ${STJ}-it-code-r-s1rl_grpo300.pt \
  ${STJ}-it-code-r-s1rl_grpo350.pt ${STJ}-it-code-r-s1rl_grpo_final.pt)
B3=$(sbatch --parsable slurm/bench_code_zoo_h100.sh humaneval \
  ${SDP}-it-code-r_grpo50.pt ${SDP}-it-code-r_grpo100.pt \
  ${SDP}-it-code-r_grpo150.pt ${SDP}-it-code-r_grpo200.pt \
  ${SDP}-it-code-r_grpo250.pt ${SDP}-it-code-r_grpo300.pt \
  ${SDP}-it-code-r_grpo350.pt ${SDP}-it-code-r-s1rl_grpo50.pt \
  ${SDP}-it-code-r-s1rl_grpo100.pt ${SDP}-it-code-r-s1rl_grpo150.pt)
B4=$(sbatch --parsable slurm/bench_code_zoo_h100.sh humaneval \
  ${SDP}-it-code-r-s1rl_grpo200.pt ${SDP}-it-code-r-s1rl_grpo250.pt \
  ${SDP}-it-code-r-s1rl_grpo300.pt ${SDP}-it-code-r-s1rl_grpo350.pt \
  ${SDP}-it-code-r-s1rl_grpo_final.pt ${R3}_grpo50.pt ${R3}_grpo100.pt \
  ${R3}_grpo150.pt ${R3}_grpo200.pt ${R3}_grpo250.pt ${R3}_grpo350.pt)

echo "tierA ${A1} ${A2} ${A3}"
echo "tierB ${B1} ${B2} ${B3} ${B4}"
