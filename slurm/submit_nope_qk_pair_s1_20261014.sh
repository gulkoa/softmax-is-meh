#!/bin/bash
# P2 seed replication (plan 2026-08-02, retargeted 2026-10-14 to the
# MATCHED-RECIPE pair): re-pretrain the 355M NoPE + QK-norm pair at seed 1,
# both arms at lr 6e-4, byte-identical configs to the s0 runs
# (nope_qk_stj_h100.sh / nope_qk_twin_h100.sh) plus --seed 1. Replicates the
# matched-ID tie (11.93 vs 11.90), softmax+qk stability at 6e-4, and the
# length dissociation under one new foundation seed.
# Self-resuming 16 h chunks; FINAL-guarded chunks no-op (insurance).
#   bash slurm/submit_nope_qk_pair_s1_20261014.sh
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm
chain () {  # script n_chunks
  local prev=""
  for i in $(seq 1 "$2"); do
    if [ -z "$prev" ]; then jid=$(sbatch --parsable "$1" --seed 1)
    else jid=$(sbatch --parsable --dependency=afterany:"$prev" "$1" --seed 1); fi
    echo "  chunk $i: job $jid${prev:+ (afterany:$prev)}"; prev="$jid"
  done
  echo "  tail: $prev"
}
echo "stj arm (nope + qk-norm, 6e-4, seed 1): 5 x 16h"; chain "$S/nope_qk_stj_h100.sh" 5
echo "sdpa arm (nope + qk-norm, 6e-4, seed 1): 4 x 16h"; chain "$S/nope_qk_twin_h100.sh" 4
