#!/bin/bash
#SBATCH --job-name=why-norm-helps
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/why-norm-helps-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/why-norm-helps-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python tmp/why_norm_helps.py \
  --qs 2.0 4.0 \
  --seeds 0 1 2 \
  --steps 3000 \
  --id-len 16 \
  --ood-lens 32 64 128 256 512 1024 2048 \
  --out "softmax-is-meh/results/why_norm_helps_${SLURM_JOB_ID}.json"

echo "DONE"
