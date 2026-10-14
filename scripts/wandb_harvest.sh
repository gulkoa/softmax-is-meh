#!/bin/bash
# Harvest: push every un-synced offline wandb run dir to wandb.
# `--append` merges multi-chunk trainings (same run id) into ONE run, in
# any order; it is a no-op for single-run dirs. Marks a dir harvest.synced once
# synced and quiet for 2 h, so later harvests skip it.
#   bash scripts/wandb_harvest.sh            # all unsynced
#   bash scripts/wandb_harvest.sh <dir> ...  # specific dirs
set -uo pipefail
W=/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/wandb
WDIR=/users/PAS2402/alexg/softmax/wandb
if [ $# -gt 0 ]; then dirs=("$@"); else dirs=("$WDIR"/offline-run-*); fi
n=0
for d in "${dirs[@]}"; do
  [ -d "$d" ] || continue
  ls "$d"/*.synced >/dev/null 2>&1 && continue
  echo "== $(basename "$d")"
  # quiet = nothing written for 2 h BEFORE we sync (the sync itself touches
  # the dir); a chunk still writing its offline run is re-synced next time
  quiet=0; [ -z "$(find "$d" -newermt '-2 hours' -print -quit)" ] && quiet=1
  out=$("$W" sync --entity gulkoa --append "$d" 2>&1); echo "$out" | grep -E "Syncing|ERROR|Error" | tail -2
  if echo "$out" | grep -q "done\." && [ "$quiet" = 1 ]; then
    touch "$d/harvest.synced"
  fi
  n=$((n+1))
done
echo "synced $n dir(s)"
