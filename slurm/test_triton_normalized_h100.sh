#!/bin/bash
#SBATCH --job-name=test-triton-norm
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/test-triton-norm-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/test-triton-norm-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

echo "########## PART 1: unit correctness suite ##########"
# Don't abort the job on unit-test failures — PART 2 (end-to-end training) is
# the ground-truth validation and must run regardless.
uv run --project softmax-is-meh/triton --no-sync python tmp/test_triton_normalized.py \
  || echo ">>> PART 1 reported failures (see above); continuing to PART 2"

echo ""
echo "########## PART 2: architecture_new end-to-end ##########"
uv run --project softmax-is-meh/triton --no-sync python tmp/flashnorm_archnew.py \
  --qs 2.0 4.0 \
  --num-iter 20 \
  --steps 3000 \
  --id-len 16 \
  --ood-lens 32 64 128 256 512 1024 \
  --out "softmax-is-meh/results/flashnorm_archnew_${SLURM_JOB_ID}.json"

echo "DONE"
