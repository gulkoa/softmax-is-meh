#!/bin/bash
# Chain submitter for the symmetric 355M rerun (stj-nope + qk-norm @6e-4).
# 5 x 16h self-resuming chunks (~60 GPU-h needed at ~73k tok/s; the 5th
# chunk is insurance and no-ops on FINAL). Each chunk depends afterany on
# the previous one. USER-GATED: run only after an explicit go.
#   bash slurm/submit_nope_qk_stj_20260904.sh [n_chunks]
set -euo pipefail
N="${1:-5}"
SCRIPT=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/nope_qk_stj_h100.sh
prev=""
for i in $(seq 1 "$N"); do
  if [ -z "$prev" ]; then
    jid=$(sbatch --parsable "$SCRIPT")
  else
    jid=$(sbatch --parsable --dependency=afterany:"$prev" "$SCRIPT")
  fi
  echo "chunk $i: job $jid${prev:+ (afterany:$prev)}"
  prev="$jid"
done
echo "tail job: $prev  (queue length eval + record creation behind it)"
