#!/bin/bash
# ONE-SHOT submission of the entire post-nope pipeline as a SLURM
# dependency ladder (2026-08-04, single-agent revival of the roadmap).
# Run ONCE from the login node: bash slurm/submit_postnope_pipeline.sh
#
# Everything hangs off TAIL = the last queued sdpa-nope chunk. FINAL-
# guarded chunks drain in minutes once sdpa hits 28,610, then:
#   at drain : #37 length evals x2, gen-entropy x2, H_eff confirm,
#              RoPE smoke (branch trainer validation), #36 round-3 RL
#   after RL : #38 P1 seed-replication pair (structural parity),
#              RoPE 355M chain (18 x 2h chunks, self-resuming)
# Total new spend ~= 5h evals + 8h RL + 2x8h P1 + ~30h RoPE, all <=355M.

set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh

TAIL=13246272
FW=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
STJB=${FW}/ckpt_gpt2-stj-q4-medium-mix_s0
SDPB=${FW}/ckpt_gpt2-sdpa-medium-mix_s0
ALPHAS="0.25,0.3,0.35,0.5,1.0"

echo "== namespaced RL bases (cp -n: no-ops if already present) =="
cp -n ${STJB}-it-code-r.pt ${FW}/ckpt_gpt2-stj-r3judge_s0.pt
cp -n ${STJB}-it-code-r.pt ${STJB}-it-code-r-s1rl.pt
cp -n ${SDPB}-it-code-r.pt ${SDPB}-it-code-r-s1rl.pt

echo "== drain-time evals =="
E1=$(sbatch --parsable -t 01:00:00 -d afterany:${TAIL} slurm/code_rl_stage_h100.sh \
  eval_longctx_scaled.py ${FW}/ckpt_gpt2-stj-q4-nope-nope-mix-lr0.0006_s0.pt --alpha ${ALPHAS})
E2=$(sbatch --parsable -t 01:00:00 -d afterany:${TAIL} slurm/code_rl_stage_h100.sh \
  eval_longctx_scaled.py ${FW}/ckpt_gpt2-sdpa-nope-nope-mix-lr0.0003_s0.pt --alpha ${ALPHAS})
E3=$(sbatch --parsable -t 01:00:00 -d afterany:${TAIL} slurm/code_rl_stage_h100.sh \
  eval_gen_entropy_stilt.py --arm stj --wandb --init-from ${STJB}.pt \
  --ckpt pretrain=${STJB}.pt --ckpt it=${STJB}-it.pt \
  --ckpt it-code-r=${STJB}-it-code-r.pt \
  --ckpt rl100=${STJB}-it-code-r_grpo100.pt \
  --ckpt rl300=${STJB}-it-code-r_grpo300.pt \
  --out results/gen_entropy_stj.json)
E4=$(sbatch --parsable -t 01:00:00 -d afterany:${TAIL} slurm/code_rl_stage_h100.sh \
  eval_gen_entropy_stilt.py --arm sdpa --wandb --init-from ${SDPB}.pt \
  --ckpt pretrain=${SDPB}.pt --ckpt it=${SDPB}-it.pt \
  --ckpt it-code-r=${SDPB}-it-code-r.pt \
  --ckpt rl100=${SDPB}-it-code-r_grpo100.pt \
  --ckpt rl300=${SDPB}-it-code-r_grpo300.pt \
  --out results/gen_entropy_sdpa.json)
E5=$(sbatch --parsable -d afterany:${TAIL} slurm/heff_confirm_h100.sh)

echo "== RoPE smoke (branch trainer validation, ~95 steps) =="
SMK=$(sbatch --parsable -t 00:40:00 -d afterany:${TAIL} slurm/gpt2_rope_h100.sh stj \
  --tag rope-smoke --total-tokens 5e7 --val-every 40 --ckpt-every 40)

echo "== #36 round-3 code-RL (fused judge, frozen cmd from exp record) =="
R3=$(sbatch --parsable -d afterany:${TAIL} slurm/grpo_code2_h100.sh \
  ${FW}/ckpt_gpt2-stj-r3judge_s0.pt \
  --solvable ${STJB}-it-code-r_mbpp_probe.json \
  --synthetic /users/PAS2402/alexg/softmax/softmax-is-meh/results/synth_code_tasks_v1.json \
  --judge-weight 0.3 --max-new 380 --steps 400 --micro-bs 8)

echo "== #38 P1 seed-replication pair (after round-3; parity: queued together) =="
P1S=$(sbatch --parsable -d afterany:${R3} slurm/grpo_code2_h100.sh \
  ${STJB}-it-code-r-s1rl.pt --solvable ${STJB}-it-code-r_mbpp_probe.json \
  --seed 1 --steps 400 --micro-bs 8)
P1T=$(sbatch --parsable -d afterany:${R3} slurm/grpo_code2_h100.sh \
  ${SDPB}-it-code-r-s1rl.pt --solvable ${SDPB}-it-code-r_mbpp_probe.json \
  --seed 1 --steps 400 --micro-bs 8)

echo "== RoPE 355M chain (after round-3 AND smoke; 18 x 2h, self-guarded) =="
PREV=$(sbatch --parsable -d afterany:${R3}:${SMK} slurm/gpt2_rope_h100.sh stj)
CHAIN="${PREV}"
for i in $(seq 2 18); do
  PREV=$(sbatch --parsable -d afterany:${PREV} slurm/gpt2_rope_h100.sh stj)
  CHAIN="${CHAIN} ${PREV}"
done

echo ""
echo "PIPELINE MAP"
echo "  longctx  stj=${E1} sdpa=${E2}"
echo "  gen-ent  stj=${E3} sdpa=${E4}"
echo "  heff     ${E5}"
echo "  smoke    ${SMK}"
echo "  round3   ${R3}"
echo "  P1       stj=${P1S} sdpa=${P1T}"
echo "  rope     ${CHAIN}"
