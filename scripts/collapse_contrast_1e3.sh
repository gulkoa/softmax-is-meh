#!/bin/bash
# Collapse signature (pre-registered, exploratory) for the 1e-3 contrast at
# seeds 2-4 (thesis/experiments/2026-10-18-1e-3-contrast-seeds-124m.md):
# every 1,000-step snapshot that exists and each run's checkpoint (the final
# one, or the last saved for a run that stopped early), on the login node
# CPU as in the survey. Run once every chain has ended. Missing files are
# listed, not fatal. Output: results/qk_collapse_contrast_1e3_<date>.txt,
# read by scripts/score_contrast_1e3.py --signature.
set -uo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh
D=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt
OUT=$S/results/qk_collapse_contrast_1e3_$(date +%Y%m%d).txt
ck=()
miss=()
for stem in stj-q4-nope-lrsweep sdpa-nope-lrsweepls; do
    for s in 2 3 4; do
        base=$D/ckpt_gpt2-$stem-lr0.001_s$s
        for st in 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
            if [ -f "$base.step$st.pt" ]; then ck+=("$base.step$st.pt"); else miss+=("$base.step$st.pt"); fi
        done
        if [ -f "$base.pt" ]; then ck+=("$base.pt"); else miss+=("$base.pt"); fi
    done
done
{
    date +%T
    git -C $S log --oneline -1 | cut -c1-7
    for f in "${miss[@]}"; do echo "missing $f"; done
    nice -n 10 $S/triton/.venv/bin/python $S/scripts/probe_locate.py "${ck[@]}" --signature --threads 16
    date +%T
} > "$OUT" 2>&1
echo "wrote $OUT (${#ck[@]} checkpoints, ${#miss[@]} missing)"
grep "^missing\|^SIGNATURE" "$OUT"
