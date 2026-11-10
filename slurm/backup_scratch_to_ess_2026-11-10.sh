#!/bin/bash
#SBATCH --job-name=stilt-backup
#SBATCH --account=PAS2836
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/backup-ess-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/backup-ess-%j.err
# Backup of 2026-11-10 (user: "assume all your scratch could be deleted
# tomorrow; back everything important up"). Copies, most important first,
# from /fs/scratch/PAS2836/alexg and from home into the project space
# /fs/ess/PAS2402/alexg/stilt-backup (backed up by OSC, not purged). Each
# stage is an rsync of explicit paths; a stage that fails does not stop the
# later ones; the exit code is the worst rsync exit code. The last stage is
# a dry-run verification that lists anything still different.
#   stage 1  the three held A@1e-3 checkpoints (+ .bak), the 35 snapshots of
#            the 1e-3 contrast runs, and every final checkpoint
#   stage 2  the GPT-2-tokenized data the queued chains read (fineweb_edu_10bt
#            shards, finemath_4plus, codeparrot_py), tokenizers, helper logs,
#            the scratch hf_cache
#   stage 3  the 138 analysed lr-sweep and probe snapshots
#   stage 4  the four tok32k shard sets
#   stage 5  home: ~/ckpt_backup and the synthetic-task result directories
#   stage 6  verification (rsync --dry-run --itemize-changes)
# Record: thesis/experiments/2026-11-10-backup-scratch-to-ess.md
set -u
S=/fs/scratch/PAS2836/alexg
E=/fs/ess/PAS2402/alexg/stilt-backup
H=/users/PAS2402/alexg
R=$H/softmax/softmax-is-meh
L=$E/.lists
RS="rsync -aW --info=stats1"
worst=0
note() { echo "=== $(date +%F_%T) $*"; }
run() { "$@"; local rc=$?; echo "rc=$rc"; [ "$rc" -gt "$worst" ] && worst=$rc; return 0; }

mkdir -p "$E/fineweb_edu_10bt" "$L" "$E/home"
cd "$S/fineweb_edu_10bt" || exit 1
ls -1 ckpt_*lr0.001_s[234].pt ckpt_*lr0.001_s[234].pt.bak ckpt_*lr0.001_s[234].step*.pt > "$L/stage1_contrast.txt"
ls -1 ckpt_*.pt | grep -v -E '\.step|\.corrupt' > "$L/stage1_finals.txt"
ls -1 ckpt_*.step*.pt | grep -v 'lr0.001_s[234]' > "$L/stage3_snapshots.txt"
echo "lists: contrast $(wc -l < "$L/stage1_contrast.txt"), finals $(wc -l < "$L/stage1_finals.txt"), other snapshots $(wc -l < "$L/stage3_snapshots.txt")"

note "stage 1 held A + contrast snapshots"
run $RS --files-from="$L/stage1_contrast.txt" "$S/fineweb_edu_10bt/" "$E/fineweb_edu_10bt/"
note "stage 1 finals"
run $RS --files-from="$L/stage1_finals.txt" "$S/fineweb_edu_10bt/" "$E/fineweb_edu_10bt/"

note "stage 2 data read by the queued chains"
run $RS --exclude='ckpt_*' "$S/fineweb_edu_10bt/" "$E/fineweb_edu_10bt/"
for d in finemath_4plus codeparrot_py tokenizers helper_logs hf_cache; do
    note "stage 2 $d"
    run $RS "$S/$d/" "$E/$d/"
done

note "stage 3 analysed snapshots"
run $RS --files-from="$L/stage3_snapshots.txt" "$S/fineweb_edu_10bt/" "$E/fineweb_edu_10bt/"

for d in fineweb_edu_10bt_tok32k finemath_4plus_tok32k codeparrot_py_tok32k fineweb_100bt_tok32k; do
    note "stage 4 $d"
    run $RS "$S/$d/" "$E/$d/"
done

note "stage 5 home ckpt_backup"
run $RS "$H/ckpt_backup/" "$E/home/ckpt_backup/"
mkdir -p "$E/home/results"
for d in "$R"/results/bfs_* "$R"/results/sorting_* "$R"/results/needle_* "$R"/results/max_* "$R"/results/binary_search_*; do
    [ -d "$d" ] || continue
    note "stage 5 $(basename "$d")"
    run $RS "$d/" "$E/home/results/$(basename "$d")/"
done

note "stage 6 verification (anything listed below is still different)"
diffs=0
check() { local n; n=$(rsync -aWn --itemize-changes "$@" | grep -v -E '^\.d' | tee -a "$L/verify_diffs.txt" | wc -l); diffs=$((diffs + n)); echo "  $n differences: $2"; }
: > "$L/verify_diffs.txt"
check --files-from="$L/stage1_contrast.txt" "$S/fineweb_edu_10bt/" "$E/fineweb_edu_10bt/"
check --files-from="$L/stage1_finals.txt" "$S/fineweb_edu_10bt/" "$E/fineweb_edu_10bt/"
check --exclude='ckpt_*' "$S/fineweb_edu_10bt/" "$E/fineweb_edu_10bt/"
check --files-from="$L/stage3_snapshots.txt" "$S/fineweb_edu_10bt/" "$E/fineweb_edu_10bt/"
for d in finemath_4plus codeparrot_py tokenizers helper_logs hf_cache fineweb_edu_10bt_tok32k finemath_4plus_tok32k codeparrot_py_tok32k fineweb_100bt_tok32k; do
    check "$S/$d/" "$E/$d/"
done
check "$H/ckpt_backup/" "$E/home/ckpt_backup/"
for d in "$R"/results/bfs_* "$R"/results/sorting_* "$R"/results/needle_* "$R"/results/max_* "$R"/results/binary_search_*; do
    [ -d "$d" ] || continue
    check "$d/" "$E/home/results/$(basename "$d")/"
done
note "sizes"
du -sh "$E"/* "$E"/home/* 2>/dev/null
note "done: worst rsync rc=$worst, verification differences=$diffs"
exit "$worst"
