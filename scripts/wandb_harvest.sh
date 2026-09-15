#!/bin/bash
# Harvest: push every un-synced offline wandb run dir to wandb.
# `--append` merges multi-chunk trainings (same run id) into ONE run, in
# any order; it is a no-op for single-run dirs. Marks each dir .synced.
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
  "$W" sync --entity gulkoa --append "$d" 2>&1 | grep -E "Syncing|ERROR|Error" | tail -2
  n=$((n+1))
done
echo "synced $n dir(s)"
