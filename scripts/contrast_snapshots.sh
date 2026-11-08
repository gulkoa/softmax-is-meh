#!/bin/bash
# Keep a 1000-step snapshot keeper (scripts/snapshot_ckpts.py --keep-every
# 1000) alive for each of the three held A@1e-3 contrast chains until its last
# chunk has ended. The keepers copy every checkpoint save whose step is a
# multiple of 1000 to <ckpt>.step<N>.pt; the pre-registered collapse signature
# (scripts/score_contrast_1e3.py --signature, run only at all-six FINAL) reads
# those copies. Written 2026-11-08: the keepers that ran since 10-19 lived in a
# session scratchpad under /tmp and were lost, with every other helper, when the
# login node rebooted on 11-05 (the cancellers loop was lost the same way on
# 10-27). A login-node reboot kills this wrapper too; restart it after one.
#   bash scripts/contrast_snapshots.sh        # the three A chains (LS done 10-21)
set -u
ROOT=/users/PAS2402/alexg/softmax/softmax-is-meh
PY=$ROOT/triton/.venv/bin/python
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
keeper() {
    local label=$1 ckpt=$2 last=$3
    "$PY" "$ROOT/scripts/snapshot_ckpts.py" "$ckpt" "$last" --keep-every 1000 2>&1 | sed "s/^/[$label] /"
    echo "[$label] $(date +%T) keeper exited"
}
keeper A@1e-3-s2 "$D/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s2.pt" 15669640 &
keeper A@1e-3-s3 "$D/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s3.pt" 15669645 &
keeper A@1e-3-s4 "$D/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s4.pt" 15669650 &
wait
echo "$(date +%T) all keepers exited"
