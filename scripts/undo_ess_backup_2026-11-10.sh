#!/bin/bash
# Undo of the 2026-11-10 ESS backup: the user asked, while job 16531778
# was copying, not to back up to other directories on this machine but to
# Hugging Face or GitHub instead. The job was cancelled (sacct: CANCELLED
# after 11 min, stages 1 and 2 complete, stage 3 partial) and this script
# removes what it had copied. Explicit path; nothing on scratch or in home
# is touched. The parent /fs/ess/PAS2402/alexg is removed only if empty.
#   bash scripts/undo_ess_backup_2026-11-10.sh 2>&1 | tee results/undo-ess-backup-2026-11-10.log
set -u
E=/fs/ess/PAS2402/alexg/stilt-backup
echo "undo start $(date)"
if squeue -h -j 16531778 2>/dev/null | grep -q .; then
    echo "job 16531778 still in the queue; refusing"; exit 1
fi
if [ -d "$E" ]; then
    du -sh -- "$E"
    rm -rf -- "$E"
    echo "removed  $E"
else
    echo "missing  $E"
fi
rmdir /fs/ess/PAS2402/alexg 2>/dev/null && echo "removed  /fs/ess/PAS2402/alexg (empty)"
ls -ld /fs/ess/PAS2402/alexg 2>&1
echo "undo end $(date)"
