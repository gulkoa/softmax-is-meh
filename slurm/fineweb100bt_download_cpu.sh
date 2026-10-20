#!/bin/bash
#SBATCH --job-name=fw100-dl
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/fw100-dl-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/fw100-dl-%j.err
# stilt.1.2 data (plan amendment 2026-10-20; record
# thesis/experiments/2026-10-20-fineweb-100bt-tok32k.md): FineWeb sample-100BT
# parquet (150 files, 302.6 GB) at a pinned revision into scratch, every file's
# size checked. HF_HOME on scratch (home is near its quota); the token is read
# from its file, never printed. tok32k worktree script.
set -euo pipefail
cd /users/PAS2402/alexg/softmax
export HF_HOME=/fs/scratch/PAS2836/alexg/hf_cache
export HF_TOKEN_PATH=/users/PAS2402/alexg/.cache/huggingface/token
uv run --project softmax-is-meh/triton --no-sync python \
  /users/PAS2402/alexg/softmax/.worktrees/tok32k/triton/dev_scripts/prepare_fineweb_tok32k.py download \
  --out /fs/scratch/PAS2836/alexg/fineweb_100bt_parquet \
  --revision 9bb295ddab0e05d785b879661af7260fed5140fc --workers 16
echo "ALL DONE"
