#!/bin/bash
#SBATCH --job-name=convert-many-hf
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/convert-many-hf-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/convert-many-hf-%j.err
# Convert and verify several research checkpoints in one GPU allocation
# (written 2026-10-30 for the research-checkpoint upload; the per-model
# convert_gpt2_hf_h100.sh is unchanged). Reads LIST lines
# "<ckpt> <staging> <trainer dev_scripts dir> <repo>" from FIRST to LAST
# (1-based, inclusive), seeds each staging dir with the HF template files
# (the README is already there from scripts/hf_research_cards.py), runs
# triton/dev_scripts/convert_verify_gpt2_hf.py with STILT_TRAINER set to
# the line's trainer, keeps its full output in <staging>/convert.log, and
# prints one summary line per checkpoint: "VERIFIED <repo>" or "FAILED <repo>".
#   sbatch slurm/convert_many_hf_h100.sh LIST FIRST LAST
set -u
LIST=$1; FIRST=$2; LAST=$3
R=/users/PAS2402/alexg/softmax/softmax-is-meh
PY=$R/triton/.venv/bin/python
export HF_HOME=/fs/scratch/PAS2836/alexg/hf_cache
export HF_HUB_OFFLINE=1
echo "node $(hostname) $(date '+%F %T') lines $FIRST-$LAST of $LIST"
sed -n "${FIRST},${LAST}p" "$LIST" | while read -r CKPT STAGING TRAINER REPO; do
    [ -n "$CKPT" ] || continue
    mkdir -p "$STAGING"
    cp "$R"/triton/hf_template/{modeling_stieltjes_gpt2.py,configuration_stieltjes_gpt2.py,handler.py,requirements.txt} "$STAGING"/
    t0=$(date +%s)
    if STILT_TRAINER="$TRAINER" $PY "$R/triton/dev_scripts/convert_verify_gpt2_hf.py" "$CKPT" "$STAGING" > "$STAGING/convert.log" 2>&1 \
       && grep -q "^VERIFIED" "$STAGING/convert.log"; then
        echo "VERIFIED $REPO ($(( $(date +%s) - t0 ))s) $(grep -E '^A: ' "$STAGING/convert.log" | cut -c1-80)"
    else
        echo "FAILED $REPO ($(( $(date +%s) - t0 ))s): $(tail -n 1 "$STAGING/convert.log" | cut -c1-120)"
    fi
done
echo "done $(date '+%F %T')"
