#!/bin/bash
#SBATCH --job-name=nl-mqar-suite
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/nl-mqar-suite-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/nl-mqar-suite-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

echo "########## PART 1: code-LM on The Stack slice (byte-level, NoPE, 3 arms) ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py --data stack --attn sdpa \
  || echo ">>> stack sdpa arm failed; continuing"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py --data stack --attn flashn --q 4 \
  || echo ">>> stack flashn-q4 arm failed; continuing"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py --data stack --attn flashn --q 2 \
  || echo ">>> stack flashn-q2 arm failed; continuing"

echo ""
echo "########## PART 1b: sanity arms on tinyshakespeare ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py --data shakespeare --attn sdpa \
  || echo ">>> shakespeare sdpa arm failed; continuing"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py --data shakespeare --attn flashn --q 4 \
  || echo ">>> shakespeare flashn-q4 arm failed; continuing"

echo ""
echo "########## PART 2: zoology MQAR (softmax vs flashn, trimmed sweep) ##########"
export PYTHONPATH=/users/PAS2402/alexg/softmax/zoology:/users/PAS2402/alexg/softmax/softmax-is-meh/triton/dev_scripts
SWEEP_DMODELS=64 SWEEP_NLRS=2 \
uv run --project softmax-is-meh/triton --no-sync python -m zoology.launch \
  softmax-is-meh/triton/dev_scripts/zoology_mqar_stieltjes.py

echo "ALL DONE"
