#!/bin/bash
#SBATCH --job-name=prepare-mathcode
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/prepare-mathcode-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/prepare-mathcode-%j.err

# 355M mixed-corpus data: FineMath (math) + Stack-Edu Python (code) ->
# GPT-2-BPE uint16 shards on scratch, capped for the target mix
# (~15B web from the existing fineweb shards + ~7B math + ~4B code).

set -euo pipefail
cd /users/PAS2402/alexg/softmax

echo "########## FineMath 4+ (cap 7B tokens) ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/prepare_hf_corpus.py \
  --dataset HuggingFaceTB/finemath --name finemath-4plus \
  --out /fs/scratch/PAS2836/alexg/finemath_4plus --max-tokens 7e9 \
  || echo ">>> finemath failed; continuing"

echo ""
echo "########## Stack-Edu Python (cap 4B tokens) ##########"
uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/prepare_hf_corpus.py \
  --dataset HuggingFaceTB/stack-edu --name python \
  --out /fs/scratch/PAS2836/alexg/stackedu_python --max-tokens 4e9 \
  || echo ">>> stack-edu failed (may be gated); trying the-stack-smol"

if [ ! -f /fs/scratch/PAS2836/alexg/stackedu_python/meta.json ]; then
  uv run --project softmax-is-meh/triton --no-sync python \
    softmax-is-meh/triton/dev_scripts/prepare_hf_corpus.py \
    --dataset bigcode/the-stack-smol --name default --split train \
    --text-key content \
    --out /fs/scratch/PAS2836/alexg/stack_smol --max-tokens 4e9 \
    || echo ">>> code corpus unavailable; proceeding web+math only"
fi

echo "ALL DONE"
