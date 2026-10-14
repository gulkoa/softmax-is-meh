#!/bin/bash
#SBATCH --job-name=retok32k
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/retok32k-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/retok32k-%j.err
# Re-tokenize every GPT-2 shard (web 40, math 26, code 20) with
# stilt-tok-32k-v1 into <src>_tok32k/ — offline, same documents, same shard
# indices, so the last shard stays the validation shard. tok32k worktree.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export HF_HUB_OFFLINE=1
TOK=${1:-/fs/scratch/PAS2836/alexg/tokenizers/stilt-tok-32k-v1}
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/retokenize_shards.py \
  --tokenizer "$TOK" --tag tok32k --workers 24 --threads 4
echo "ALL DONE"
