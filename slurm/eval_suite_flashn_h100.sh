#!/bin/bash
#SBATCH --job-name=eval-suite-flashn
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/eval-suite-flashn-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/eval-suite-flashn-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax

echo "########## 1/3 QUALITY: dtype correctness + num_iter sensitivity ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/quality_numiter_dtypes.py \
  || echo ">>> quality suite reported failures; continuing"

echo ""
echo "########## 2/3 SPEED: causal + dtypes longctx benchmark ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/bench_causal_dtypes.py \
  || echo ">>> speed benchmark reported failures; continuing"

echo ""
echo "########## 3/3 PERFORMANCE: multi-seed max-retrieval ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/multiseed_maxretr_flashn.py

echo "ALL DONE"
