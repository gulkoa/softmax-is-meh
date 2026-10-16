#!/bin/bash
#SBATCH --job-name=kspeed-real
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-speed-real-identity-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-speed-real-identity-%j.err
# Job 5 of thesis/experiments/2026-10-16-kernel-causal-skip.md:
# 1. old (dc8dd3d) vs branch timings at N = 1,024 (355M / 124M shapes) and
#    N = 4,096 / 8,192 (old was never timed at long N);
# 2. one real micro-step (validation tokens, bf16 autocast) through both
#    kernels on high-score sweep snapshots (A@2e-3 step 4,000, M 1,440;
#    A@4e-3 step 3,000, M 712): loss and every parameter gradient bit for bit.
# Snapshots are read-only step copies; nothing is written outside results/.
set -uo pipefail
W=/users/PAS2402/alexg/softmax/.worktrees/kernel-speed
PY=/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
cd $W/triton
nvidia-smi --query-gpu=name --format=csv,noheader
echo "== 1. identity (random inputs) + timings, old vs branch"
$PY $W/triton/test_causal_skip.py --bench; echo "bench exit $?"
echo "== 2. real micro-step identity on high-score snapshots"
$PY $W/triton/check_real_identity.py \
    $D/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s0.step4000.pt \
    $D/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.004_s0.step3000.pt; echo "real exit $?"
echo "ALL DONE"
