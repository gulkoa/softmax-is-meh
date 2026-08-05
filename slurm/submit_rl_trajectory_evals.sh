#!/bin/bash
# ONE-SHOT (2026-08-05): held-out MBPP trajectory evals for round-3 and
# the P1 seed-replication pair (checkpoint selection is ALWAYS by
# held-out pass@k, never training curves), plus the kernel-branch
# regression-test job. All 1h walls; they age behind the nope-it SFT.

set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt

for CK in ${FW}/ckpt_gpt2-stj-r3judge_s0_grpo100.pt \
          ${FW}/ckpt_gpt2-stj-r3judge_s0_grpo200.pt \
          ${FW}/ckpt_gpt2-stj-r3judge_s0_grpo300.pt \
          ${FW}/ckpt_gpt2-stj-r3judge_s0_grpo_final.pt \
          ${FW}/ckpt_gpt2-stj-q4-medium-mix_s0-it-code-r-s1rl_grpo100.pt \
          ${FW}/ckpt_gpt2-stj-q4-medium-mix_s0-it-code-r-s1rl_grpo200.pt \
          ${FW}/ckpt_gpt2-stj-q4-medium-mix_s0-it-code-r-s1rl_grpo300.pt \
          ${FW}/ckpt_gpt2-stj-q4-medium-mix_s0-it-code-r-s1rl_grpo_final.pt \
          ${FW}/ckpt_gpt2-sdpa-medium-mix_s0-it-code-r-s1rl_grpo100.pt \
          ${FW}/ckpt_gpt2-sdpa-medium-mix_s0-it-code-r-s1rl_grpo200.pt \
          ${FW}/ckpt_gpt2-sdpa-medium-mix_s0-it-code-r-s1rl_grpo300.pt \
          ${FW}/ckpt_gpt2-sdpa-medium-mix_s0-it-code-r-s1rl_grpo_final.pt; do
  J=$(sbatch --parsable -t 01:00:00 slurm/code_rl_stage_h100.sh \
      eval_code_stilt.py "${CK}" --dataset mbpp --limit 300)
  echo "eval ${J}  $(basename ${CK})"
done

J=$(sbatch --parsable slurm/kernel_tests_h100.sh)
echo "kernel-tests ${J}"
