#!/bin/bash
#SBATCH --job-name=train-tok32k
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/train-tok32k-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/train-tok32k-%j.err
# Train stilt-tok-32k-v1 on 1B GPT-2 tokens of the 70/15/15 mix (decoded from
# the train shards, offline). Runs the tok32k worktree script.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export HF_HUB_OFFLINE=1 RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
OUT=${1:-/fs/scratch/PAS2836/alexg/tokenizers/stilt-tok-32k-v1}
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/train_tokenizer.py --out "$OUT" "${@:2}"
cp "$OUT"/manifest.json "$OUT"/tokenizer.json "$OUT"/tokenizer_config.json "$OUT"/special_tokens_map.json \
   /users/PAS2402/alexg/softmax/softmax-is-meh/results/tokenizers/$(basename "$OUT")/ 2>/dev/null || true
echo "ALL DONE"
