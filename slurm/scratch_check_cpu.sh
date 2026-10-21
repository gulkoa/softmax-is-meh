#!/bin/bash
#SBATCH --job-name=scratch-check
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/scratch-check-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/scratch-check-%j.err
# CPU-only check from a compute node, written 2026-10-21 during the login
# node's stale scratch mount: is the 1e-3 contrast data directory readable
# (shard count), and do the six contrast checkpoints (+ their .bak files)
# load (step)? Read-only; loads with mmap on the CPU. No GPU.
set -u
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
PY=/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python
echo "node $(hostname) $(date '+%F %T')"
echo "mount: $(df -h /fs/scratch/PAS2836 2>&1 | tail -1)"
echo "shards: $(ls "$D"/*.npy 2>/dev/null | wc -l) npy, $(ls "$D"/*.bin 2>/dev/null | wc -l) bin"
ls -la --time-style=+%F_%T "$D" 2>&1 | grep -E "lr0.001_s[234]" 
for stem in stj-q4-nope-lrsweep sdpa-nope-lrsweepls; do
  for s in 2 3 4; do
    for f in "$D/ckpt_gpt2-$stem-lr0.001_s$s.pt" "$D/ckpt_gpt2-$stem-lr0.001_s$s.pt.bak"; do
      [ -f "$f" ] || { echo "MISSING $f"; continue; }
      st=$($PY -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)['step'])" "$f" 2>&1 | tail -1)
      echo "$(basename "$f"): step $st"
    done
  done
done
echo "done $(date '+%F %T')"
