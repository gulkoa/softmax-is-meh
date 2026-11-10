#!/bin/bash
# Cleanup of 2026-11-10 (afternoon), on the user's instruction "Remove wiki
# data dump, delete downloadable python packages". Explicit absolute paths
# only; each is printed with its size and then removed. DRY=1 prints the
# sizes without removing (and skips the slow lexemes comparison).
#
#   bash scripts/cleanup_2026-11-10_wikidata_pypkgs.sh 2>&1 | tee results/cleanup-2026-11-10-wikidata-pypkgs.log
#   DRY=1 bash scripts/cleanup_2026-11-10_wikidata_pypkgs.sh
#
# Removed, and how each comes back:
#   ~/nlp/ce/latest-all.json.bz2   Wikidata full JSON dump, file dated
#                                  2025-04-30, 95.7 GB. A current dump is at
#                                  dumps.wikimedia.org/wikidatawiki/entities/
#                                  latest-all.json.bz2; that exact snapshot
#                                  is not re-fetchable.
#   ~/nlp/ce/latest-lexemes.json   decompressed Wikidata lexemes dump, 6.0 GB;
#                                  removed only if it is byte-identical to
#                                  bzcat of the kept latest-lexemes.json.bz2
#                                  (376 MB); `bunzip2 -k latest-lexemes.json.bz2`
#                                  recreates it.
#   ~/.conda/pkgs                  conda package cache: 231 tarballs and the
#                                  extracted packages. The envs hold hard
#                                  links to the extracted files (checked at
#                                  run time: no env symlink points into the
#                                  cache), so they keep working; conda
#                                  refills the cache on the next install.
#   ~/.local/lib/python3.6, 3.7, 3.10
#                                  pip --user site-packages of the old OSC
#                                  python modules (flaml/xgboost for 3.6;
#                                  torch/transformer_lens/wandb for 3.7;
#                                  torch/lightning/bitsandbytes/wandb for
#                                  3.10). All are PyPI installs (no editable
#                                  or local installs found). The dist-info
#                                  names and versions are written to the log
#                                  before removal; `pip install --user`
#                                  brings any of them back.
#
# Not touched: ~/.local/lib/python3.9 (the uv shim and git-filter-repo),
# ~/.local/bin, ~/.local/share/uv (holds the CPython 3.14 that
# triton/.venv runs on), the conda envs ai, nlp and bench, triton/.venv
# (the held A@1e-3 chains run `uv run --no-sync` from it), ~/.cache/huggingface,
# the processed Wikidata outputs in ~/nlp/ce (wikidata_items*, objects.json,
# english-*.json, the lexemes .bz2, the notebooks), ~/ckpt_backup, the
# synthetic-task result directories, ~/INR, and everything else not listed.
set -u
H=/users/PAS2402/alexg
DRY=${DRY:-0}
total_kb=0

rm_path() {
    local p=$1 kb
    if [ ! -e "$p" ]; then
        echo "missing  $p"
        return
    fi
    kb=$(du -sk -- "$p" | cut -f1)
    total_kb=$((total_kb + kb))
    echo "$(du -sh -- "$p" | cut -f1)	$p"
    if [ "$DRY" = 1 ]; then
        return
    fi
    rm -rf -- "$p"
}

# true if any process maps or holds open something under the given prefix
in_use() {
    local p=$1
    if grep -l -F -- "$p" /proc/[0-9]*/maps 2>/dev/null | grep -q .; then
        return 0
    fi
    if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q -F -- "$p"; then
        return 0
    fi
    return 1
}

usage() {
    echo "--- $1"
    quota -s 2>/dev/null | sed -n 3,4p
    du -sh -- "$H/.conda" "$H/.local" "$H/nlp" 2>/dev/null
}

echo "cleanup 2026-11-10 (wikidata dump, python packages) start $(date) DRY=$DRY"
usage before

# 1. Wikidata full dump
rm_path "$H/nlp/ce/latest-all.json.bz2"

# 2. decompressed lexemes dump, only if identical to the kept .bz2
LX=$H/nlp/ce/latest-lexemes.json
if [ ! -e "$LX" ] || [ ! -e "$LX.bz2" ]; then
    echo "skipped  $LX (file or its .bz2 missing)"
elif [ "$DRY" = 1 ]; then
    echo "dry      $LX would be compared with bzcat $LX.bz2 first"
    rm_path "$LX"
elif bzcat -- "$LX.bz2" | cmp -s - "$LX"; then
    echo "verified $LX is identical to bzcat $LX.bz2"
    rm_path "$LX"
else
    echo "skipped  $LX (differs from bzcat $LX.bz2)"
fi

# 3. conda package cache
PK=$H/.conda/pkgs
nsym=$(find "$H/.conda/envs" -type l -lname "*/.conda/pkgs/*" 2>/dev/null | wc -l)
if [ "$nsym" != 0 ]; then
    echo "skipped  $PK ($nsym env symlinks point into it)"
elif in_use "$PK/"; then
    echo "skipped  $PK (a process uses it)"
else
    echo "checked  no env symlink points into $PK"
    rm_path "$PK"
fi

# 4. pip --user site-packages
for v in 3.6 3.7 3.10; do
    d=$H/.local/lib/python$v
    if [ ! -e "$d" ]; then
        echo "missing  $d"
        continue
    fi
    echo "--- $d dist-info (reinstall record)"
    ls -- "$d/site-packages" | grep -E "\.dist-info$" | sed 's/\.dist-info$//' | tr '\n' ' '
    echo
    if in_use "$d/"; then
        echo "skipped  $d (a process uses it)"
    else
        rm_path "$d"
    fi
done

usage after
echo "cleanup 2026-11-10 (wikidata dump, python packages) end $(date): $((total_kb / 1024 / 1024)) GiB listed$([ "$DRY" = 1 ] && echo ', nothing removed (DRY=1)' || echo ' and removed')"
