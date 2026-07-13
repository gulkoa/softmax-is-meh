#!/bin/bash
#SBATCH --job-name=as-robustness
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/as-robustness-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/as-robustness-%j.err

set -euo pipefail

cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline

echo "########## PART 1: AS d-scale/lr robustness (max-retrieval, 16k stretch) ##########"
SWEEP_DEMBS="256,512" SWEEP_QORDERS="4" SWEEP_LRS="3e-4,1e-3,3e-3" \
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/asstj_16k_stretch.py \
  || echo ">>> part 1 reported failures; continuing"

echo ""
echo "########## PART 2: code-LM AS arms (The Stack, 19M x2 seeds + 85M) ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
  --data stack --attn asflashn --q 4 --seed 0 \
  --eval-blocks 512 1024 2048 4096 8192 16384 \
  || echo ">>> 19M asflashn s0 failed; continuing"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
  --data stack --attn asflashn --q 4 --seed 1 \
  --eval-blocks 512 1024 2048 4096 8192 16384 \
  || echo ">>> 19M asflashn s1 failed; continuing"

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/nl_charlm_stieltjes.py \
  --data stack --attn asflashn --q 4 --n-layer 12 --n-head 12 --n-embd 768 \
  --eval-blocks 512 1024 2048 4096 8192 16384 \
  || echo ">>> 85M asflashn failed; continuing"

echo ""
echo "########## PART 3: MQAR AS multi-seed (q4, lr 1e-3, seeds 124/125) ##########"
export PYTHONPATH=/users/PAS2402/alexg/softmax/zoology:/users/PAS2402/alexg/softmax/softmax-is-meh/triton/dev_scripts
AS_SEEDS="124,125" AS_QORDERS="4" AS_LRS="1e-3" \
uv run --project softmax-is-meh/triton --no-sync python -m zoology.launch \
  softmax-is-meh/triton/dev_scripts/zoology_mqar_as.py

echo "ALL DONE"
