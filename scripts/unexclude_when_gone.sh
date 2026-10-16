#!/bin/bash
# Lift a slow-node exclusion once the neighbour jobs that slowed it are gone.
#   bash scripts/unexclude_when_gone.sh NODE NEIGHBOUR_JOB [NEIGHBOUR_JOB ...]
# Rule (2026-10-16): exclude a node only while its slowing neighbour is still
# there. Polls squeue every 5 min until none of the neighbour jobs is queued
# or running (two readings in a row), then removes NODE from the ExcNodeList of every pending job of
# this user (scontrol update replaces the list; queue age is kept). Prints
# each change. DRY=1 prints what it would change instead.
set -u
[ $# -ge 2 ] || { sed -n 2,9p "$0"; exit 2; }
NODE=$1; shift
NB=$(echo "$@" | tr ' ' ',')
gone=0   # two empty readings 5 min apart, so a transient squeue failure is not "gone"
while [ $gone -lt 2 ]; do
    if [ -n "$(squeue -j "$NB" -h -o %i 2>/dev/null)" ]; then gone=0; else gone=$((gone + 1)); fi
    [ $gone -lt 2 ] && sleep 300
done
# Serialize the read-modify-write of ExcNodeList: two instances lifting
# different nodes at once overwrote each other on 10-16 18:13.
exec 9>>"$HOME/.unexclude_when_gone.lock"
flock 9
echo "$(date +%T) neighbour job(s) $NB gone from the queue; lifting $NODE"
for j in $(squeue -u "$USER" -h -t PD -o %i); do
    cur=$(scontrol show job "$j" | grep -o 'ExcNodeList=[^ ]*' | cut -d= -f2)
    [ -z "$cur" ] || [ "$cur" = "(null)" ] && continue
    hosts=$(scontrol show hostnames "$cur")
    echo "$hosts" | grep -qx "$NODE" || continue
    new=$(echo "$hosts" | grep -vx "$NODE" | paste -sd,)
    if [ -n "${DRY:-}" ]; then
        echo "DRY $j: $cur -> ${new:-<empty>}"
    else
        scontrol update JobId="$j" ExcNodeList="$new" && echo "$j: $cur -> ${new:-<empty>}"
    fi
done
echo "$(date +%T) done"
