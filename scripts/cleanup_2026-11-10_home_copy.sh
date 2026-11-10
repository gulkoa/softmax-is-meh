#!/bin/bash
# Removes the hand copy of the three held A@1e-3 checkpoints (and their
# identical .bak files) made in home on 2026-11-10 before the user directed
# that backups go to GitHub or Hugging Face, not to other directories on
# this machine. Refuses unless scripts/hf_upload_trainer_state.py --verify
# --sha reports every file on the Hub. Explicit path; nothing else is touched.
#   bash scripts/cleanup_2026-11-10_home_copy.sh 2>&1 | tee results/cleanup-2026-11-10-home-copy.log
set -u
R=/users/PAS2402/alexg/softmax/softmax-is-meh
B=/users/PAS2402/alexg/ckpt_backup/held_a_1e-3_2026-11-10
echo "start $(date)"
if ! "$R/triton/.venv/bin/python" "$R/scripts/hf_upload_trainer_state.py" --verify --sha 2>&1 | grep -v -i warning | tee /dev/stderr | grep -q "0 problems"; then
    echo "Hub copy not verified; refusing"; exit 1
fi
if [ -d "$B" ]; then
    du -sh -- "$B"
    rm -rf -- "$B"
    echo "removed  $B"
else
    echo "missing  $B"
fi
echo "end $(date)"
