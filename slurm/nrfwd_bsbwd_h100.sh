#!/bin/bash
#SBATCH --job-name=nrfwd-bsbwd
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/nrfwd-bsbwd-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/nrfwd-bsbwd-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python tmp/nrfwd_bsbwd.py \
  --q "${Q:-4.0}" \
  --seed "${SEED:-0}" \
  --training-steps 5000 \
  --batch-size 256 \
  --lr 1e-3 \
  --warmup-steps 500 \
  --id-len 16 \
  --ood-lens 32 64 128 256 512 \
  --eval-samples-id 2048 \
  --eval-samples-ood 1024 \
  --d-emb 128 \
  --n-classes 10 \
  --out "softmax-is-meh/results/nrfwd_bsbwd_seed${SEED:-0}_q${Q:-4.0}_${SLURM_JOB_ID}.json"

echo "DONE seed=${SEED:-0} q=${Q:-4.0}"
