#!/bin/bash
#SBATCH --job-name=fw100-tok32k
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/fw100-tok32k-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/fw100-tok32k-%j.err
# stilt.1.2 data (plan amendment 2026-10-20; record
# thesis/experiments/2026-10-20-fineweb-100bt-tok32k.md): tokenize the FineWeb
# sample-100BT parquet with stilt-tok-32k-v1 into ~250M-token uint16 shards
# (last = validation), dropping every document of the old FineWeb-Edu
# validation shard (whitespace-stripped text match). Submit with
# --dependency=afterok on the download job. tok32k worktree script.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export HF_HUB_OFFLINE=1
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/prepare_fineweb_tok32k.py build \
  --parquet /fs/scratch/PAS2836/alexg/fineweb_100bt_parquet \
  --out /fs/scratch/PAS2836/alexg/fineweb_100bt_tok32k \
  --tokenizer /fs/scratch/PAS2836/alexg/tokenizers/stilt-tok-32k-v1 \
  --exclude-shard /fs/scratch/PAS2836/alexg/fineweb_edu_10bt_tok32k/shard_0039.bin \
  --workers 24 --threads 4
echo "ALL DONE"
