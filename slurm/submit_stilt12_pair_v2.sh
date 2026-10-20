#!/bin/bash
# Launch the stilt.1.2 355M pair with the 10-20 amendment (FineWeb sample-100BT,
# fixed 14,020,853,435-token budget): launch copy of submit_stilt12_pair.sh.
# A 15-minute debug smoke of the 355M Stieltjes config (compile/OOM) gates
# every chunk of both chains: chunk 1 afterok on the smoke, every later chunk
# afterok on the smoke AND afterany on the previous chunk. Chunks are 4 h with
# --time-min 2 h so backfill can start them (the 10-16 queue lesson). The
# plan's ~50 / ~40 GPU-h at 355M take ~15 / ~12 full 4 h chunks; 26 / 20
# still cover them if every chunk runs only ~2.5 h (minus ~20 min per chunk
# for compile and the steps lost since the last checkpoint). If FINAL is not
# logged when the last chunk is queued, append more afterany on it. A chunk
# that starts after FINAL re-saves and exits in minutes; cancel the spares
# once FINAL is logged. Every line printed is also appended to
# results/stilt12_jobs.txt, which scripts/score_stilt12.py reads (chunks
# added later: append "<stj|sdpa> chunk <i>: <jobid>" lines to it).
# USER-GATED (GPU go). Usage: bash slurm/submit_stilt12_pair_v2.sh
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/stilt12_pair_v2_h100.sh
JOBS=/users/PAS2402/alexg/softmax/softmax-is-meh/results/stilt12_jobs.txt
log() { echo "$@"; echo "$@" >> "$JOBS"; }
log "submitted $(date '+%Y-%m-%d %H:%M:%S')"
CHUNKS_STJ=${CHUNKS_STJ:-26}
CHUNKS_SDPA=${CHUNKS_SDPA:-20}
SM=$(sbatch --parsable -p debug -t 00:15:00 "$S" stj --total-tokens 2e7 --val-every 10 --ckpt-every 20 --tag stilt12smoke)
log "smoke (stj, 355M, compile): $SM"
for ARM in stj sdpa; do
  n=$CHUNKS_STJ; [ "$ARM" = sdpa ] && n=$CHUNKS_SDPA
  prev=""
  for i in $(seq 1 "$n"); do
    dep="afterok:$SM"; [ -n "$prev" ] && dep="afterok:$SM,afterany:$prev"
    jid=$(sbatch --parsable -t 4:00:00 --time-min 2:00:00 --dependency="$dep" "$S" "$ARM")
    log "$ARM chunk $i: $jid"; prev=$jid
  done
done
