#!/bin/bash
# ALiBi 124M pilot chain: 3 x 10 h self-resuming chunks (the float-mask SDPA
# path runs ~80k tok/s, so 5B tokens needs ~17 h); FINAL-guarded chunks no-op.
#   bash slurm/submit_alibi_pilot_20261014.sh
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/alibi_pilot_h100.sh
prev=""
for i in 1 2 3; do
  if [ -z "$prev" ]; then jid=$(sbatch --parsable "$S")
  else jid=$(sbatch --parsable --dependency=afterany:"$prev" "$S"); fi
  echo "chunk $i: job $jid${prev:+ (afterany:$prev)}"; prev="$jid"
done
echo "tail: $prev"
