#!/bin/bash
# Launch the stilt.1.2 355M pair: a 1-chunk GPU smoke per arm is NOT needed
# (the trainer path ran at 124M on the same shards), but each arm's first
# chunk is gated afterok on a 15-minute debug smoke of the 355M config so a
# compile/OOM failure cancels the whole chain. 5 x 16 h chunks per arm; every
# later chunk depends afterany on the previous AND afterok on the smoke.
# USER-GATED. Usage: bash slurm/submit_stilt12_pair.sh
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/stilt12_pair_h100.sh
SM=$(sbatch --parsable -p debug -t 00:15:00 "$S" stj --total-tokens 2e7 --val-every 10 --ckpt-every 20 --tag stilt12smoke)
echo "smoke (stj, 355M, compile): $SM"
for ARM in stj sdpa; do
  prev=""
  for i in 1 2 3 4 5; do
    if [ -z "$prev" ]; then jid=$(sbatch --parsable --dependency=afterok:$SM "$S" $ARM)
    else jid=$(sbatch --parsable --dependency=afterok:$SM,afterany:$prev "$S" $ARM); fi
    echo "$ARM chunk $i: $jid"; prev=$jid
  done
done
