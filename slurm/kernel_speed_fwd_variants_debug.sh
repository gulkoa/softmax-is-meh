#!/bin/bash
#SBATCH --job-name=kspeed-var
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-speed-fwd-variants-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/kernel-speed-fwd-variants-%j.err
# Job 2 of thesis/experiments/2026-10-16-kernel-causal-skip.md: why the causal
# tile skip slowed the forward. Compiles the forward old (dc8dd3d) and new
# (branch kernel-speed) across tile sizes x num_warps x num_stages at the 355M
# and 124M shapes (registers, spills, pipelining, time), then re-runs the
# bit-identity test with its raw-bits comparison.
set -uo pipefail
W=/users/PAS2402/alexg/softmax/.worktrees/kernel-speed
PY=/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python
cd $W/triton
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -c "import triton, torch; print('triton', triton.__version__, 'torch', torch.__version__)"
echo "== variants, 355M shape"
$PY $W/triton/bench_fwd_variants.py --B 8 --H 16
echo "== variants, 124M shape"
$PY $W/triton/bench_fwd_variants.py --B 16 --H 12
echo "== bit-identity (raw bits)"
$PY $W/triton/test_causal_skip.py; echo "identity exit $?"
echo "ALL DONE"
