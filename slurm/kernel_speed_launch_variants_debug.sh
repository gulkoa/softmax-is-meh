#!/bin/bash
#SBATCH --job-name=kspeed-launch
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-speed-launch-variants-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-speed-launch-variants-%j.err
# Job 3 of thesis/experiments/2026-10-16-kernel-causal-skip.md: num_warps x
# num_stages per kernel (fwd, delta, dkdv, dq) on branch kernel-speed, bit
# identity per variant, at the 355M / 124M training shapes and N = 4,096 / 8,192.
set -uo pipefail
W=/users/PAS2402/alexg/softmax/.worktrees/kernel-speed
PY=/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python
cd $W/triton
nvidia-smi --query-gpu=name --format=csv,noheader
$PY $W/triton/bench_launch_variants.py
echo "ALL DONE"
