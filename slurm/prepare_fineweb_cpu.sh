#!/bin/bash
#SBATCH --job-name=prepare-fineweb
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/prepare-fineweb-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/prepare-fineweb-%j.err

# GPT-2 project M0: FineWeb-Edu sample-10BT -> GPT-2-BPE uint16 shards on
# scratch (download ~27GB parquet + ~20GB shards).

set -euo pipefail
cd /users/PAS2402/alexg/softmax

uv run --project softmax-is-meh/triton --no-sync python \
  softmax-is-meh/triton/dev_scripts/prepare_fineweb.py

echo "ALL DONE"
