#!/bin/bash
# Tokenizer pilot pair: stj + sdpa on the E/F recipe over the stilt-tok-32k-v1
# shards, 2 x 10 h self-resuming chunks per arm, gated on the re-tokenization
# job (15497225) and the trainer GPU smoke (15497253) both succeeding.
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/tok32k_pilot_h100.sh
for ARM in stj sdpa; do
  j1=$(sbatch --parsable --dependency=afterok:15497225:15497253 "$S" $ARM)
  j2=$(sbatch --parsable --dependency=afterany:$j1 "$S" $ARM)
  echo "$ARM: $j1 -> $j2"
done
