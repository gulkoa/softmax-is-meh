#!/bin/bash
# Scratch cleanup of 2026-11-08 (user request: back everything up, then remove
# what is safe). Every path below is explicit; nothing is globbed. Removed:
#  - 27 pre-save .bak copies of finished 124M/355M checkpoints whose final .pt
#    was converted, verified and uploaded to the Hub (the three held A@1e-3
#    seeds 2-4 keep their .bak as resume safety);
#  - 7 smoke-run checkpoints (excluded from the Hub by policy, no analysis uses them);
#  - the 43 Hub staging directories (each carries an .uploaded marker; the Hub
#    holds the same files; a debug-partition convert job recreates them);
#  - the 2026-10 hf_probe_355m_nope round-trip artifact (a from-HF reload test).
# Kept: all 58 final .pt files, all 173 .step<N>.pt snapshots, the data shards,
# the parquet download. Prints each size, then removes. Run once:
#   bash scripts/cleanup_2026-11-08.sh
set -u
total=0
rm_path() {
  local p=$1
  if [ -e "$p" ]; then
    local kb; kb=$(du -sk "$p" | cut -f1); total=$((total + kb))
    echo "rm $(du -sh "$p" | cut -f1) $p"
    rm -rf -- "$p"
  else
    echo "absent $p"
  fi
}
# --- 27 .bak copies of uploaded finals
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-logitprobe-lr0.001_s1.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.0003_s0.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.0003_s1.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.0003_s2.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.0003_s3.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.0003_s4.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.0006_s0.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.0006_s1.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.001_s2.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.001_s3.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-lrsweepls-lr0.001_s4.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-qk-lrsweep-lr0.002_s0.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-qk-lrsweep-lr0.004_s0.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-qk-lrsweep-lr0.008_s0.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001_s1.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-qk-nope-mix-lr0.0006_s1.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-logitprobe-lr0.001_s1.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s0.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s1.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s2.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s3.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s4.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.004_s0.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.004_s1.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-lrsweep-lr0.008_s0.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001_s1.pt.bak"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-qk-nope-mix-lr0.0006_s1.pt.bak"
# --- 7 smoke checkpoints
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-alibi-qk-modern-nocap-alibismoke-lr0.001_s0.pt"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-logitprobesmoke-lr0.001_s1.pt"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-logitprobesmoke2-lr0.001_s1.pt"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-logitprobesmoke3-lr0.001_s1.pt"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-sdpa-nope-qk-lrsweepsmoke-lr0.002_s0.pt"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-alibi-qk-modern-nocap-stjalibismoke-lr0.001_s0.pt"
rm_path "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt/ckpt_gpt2-stj-q4-nope-logitprobesmoke-lr0.001_s1.pt"
# --- 43 uploaded staging directories
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-alibi-qk-modern-nocap-qkpilot-lr0.001-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-logitprobe-lr0.001-s1"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-lrsweepls-lr0.0003-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-lrsweepls-lr0.0003-s1"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-lrsweepls-lr0.0003-s2"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-lrsweepls-lr0.0003-s3"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-lrsweepls-lr0.0003-s4"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-lrsweepls-lr0.0006-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-lrsweepls-lr0.0006-s1"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-lrsweepls-lr0.001-s2"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-lrsweepls-lr0.001-s3"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-lrsweepls-lr0.001-s4"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qk-lrsweep-lr0.002-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qk-lrsweep-lr0.004-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qk-lrsweep-lr0.008-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-s1"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qk-qkpilot-lr0.0006-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qk-qkpilot-lr0.001-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qkpilot-lr0.0006-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qkpilot-lr0.001-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qkpilot-lr0.001-s1"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qkpilotls-lr0.001-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/sdpa-nope-qkpilotls-lr0.001-s1"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-alibi-qk-modern-nocap-qkpilot-lr0.001-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-logitprobe-lr0.001-s1"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-lrsweep-lr0.002-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-lrsweep-lr0.002-s1"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-lrsweep-lr0.002-s2"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-lrsweep-lr0.002-s3"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-lrsweep-lr0.002-s4"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-lrsweep-lr0.004-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-lrsweep-lr0.004-s1"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-lrsweep-lr0.008-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-s1"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-qk-nocap-qkpilot-lr0.0006-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-qk-nocap-qkpilot-lr0.001-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-qkpilot-lr0.0006-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-qkpilot-lr0.001-s0"
rm_path "/fs/scratch/PAS2836/alexg/hf_research_staging/stj-q4-nope-qkpilot-lr0.001-s1"
# --- probe artifact
rm_path "/fs/scratch/PAS2836/alexg/hf_probe_355m_nope"
echo "removed $((total / 1024 / 1024)) GiB in total"
