#!/bin/bash
#SBATCH --job-name=flashfull-archnew
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/flashfull-archnew-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/flashfull-archnew-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python tmp/flashfull_archnew.py \
  --q "${Q:-4.0}" \
  --num-iter "${NUM_ITER:-15}" \
  --steps "${STEPS:-3000}" \
  --id-len 16 \
  --ood-lens 32 64 128 256 \
  --out "softmax-is-meh/results/flashfull_archnew_q${Q:-4.0}_${SLURM_JOB_ID}.json"

echo "DONE q=${Q:-4.0}"
