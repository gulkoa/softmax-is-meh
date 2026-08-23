#!/bin/bash
# RoPE 355M chain refill #2 (2026-08-16). Refill #1 (13457491-502) drained
# at 24,860/28,610 — observed pace ~500 steps per 2h chunk (resume +
# data-build overhead eats ~25% of each chunk), not the ~880 estimated.
# Remaining ~3,750 steps => 9 chunks; FINAL-guarded chunks no-op in
# seconds once the trainer hits max steps.
set -euo pipefail
cd /users/PAS2402/alexg/softmax/softmax-is-meh
PREV=$(sbatch --parsable slurm/gpt2_rope_h100.sh stj)
CHAIN="${PREV}"
for i in $(seq 2 9); do
  PREV=$(sbatch --parsable -d afterany:${PREV} slurm/gpt2_rope_h100.sh stj)
  CHAIN="${CHAIN} ${PREV}"
done
echo "rope-refill2 ${CHAIN}"
