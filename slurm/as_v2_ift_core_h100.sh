#!/bin/bash
#SBATCH --job-name=as-v2-ift-core
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/as-v2-ift-core-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/as-v2-ift-core-%j.err

# Core-suite iteration: rerun the WINNING max-retrieval cells (V2
# per-position gamma) with the IFT gradient (AS_IFT=1). Question: does the
# smooth gradient improve the thesis's central seed-robust results
# (means/variance/convergence) vs the detached baseline (job 12342643)?

set -euo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

echo "########## d256 lr3e-4 x3 seeds (detached baseline: 91.4/82.5/69.3/53.7/38.4/26.5) ##########"
AS_V2=1 AS_IFT=1 SWEEP_DEMBS="256" SWEEP_QORDERS="4" SWEEP_LRS="3e-4" \
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/asstj_16k_stretch.py \
  || echo ">>> d256 failures; continuing"

echo ""
echo "########## d512 lr1e-3 x3 seeds (detached baseline: 91.9/83.5/68.8/52.0/38.6/25.3) ##########"
AS_V2=1 AS_IFT=1 SWEEP_DEMBS="512" SWEEP_QORDERS="4" SWEEP_LRS="1e-3" \
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/asstj_16k_stretch.py \
  || echo ">>> d512 failures; continuing"

echo "ALL DONE"
