#!/bin/bash
# Stage a trained checkpoint for Hugging Face: seed <staging_dir> from the
# canonical template (triton/hf_template: modeling + config + handler +
# requirements), drop in the model card, and submit the GPU convert+verify
# job (logits parity vs the trainer kernel, cached==uncached generation).
# On "VERIFIED" in the job log, publish from the login node with
#   python triton/dev_scripts/upload_hf_stilt.py <staging_dir> GulkoA/<repo>
# Usage: bash scripts/hf_stage.sh <ckpt.pt> <staging_dir> <README.md> [sbatch args...]
set -euo pipefail
CKPT=$1; STAGING=$2; README=$3; shift 3
R=/users/PAS2402/alexg/softmax/softmax-is-meh
mkdir -p "$STAGING"
cp "$R"/triton/hf_template/{modeling_stieltjes_gpt2.py,configuration_stieltjes_gpt2.py,handler.py,requirements.txt} "$STAGING"/
cp "$README" "$STAGING"/README.md
JOB=$(sbatch --parsable "$@" "$R"/slurm/convert_gpt2_hf_h100.sh "$CKPT" "$STAGING")
echo "staged $STAGING; convert+verify job $JOB (log: results/convert-gpt2-hf-$JOB.out)"
