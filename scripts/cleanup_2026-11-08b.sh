#!/bin/bash
# Home cleanup of 2026-11-08, part b (user request: back up, then remove what
# is safe). Removes the 13 staging directories of the published 355M Stilt
# models under results/ (gitignored safetensors + card + handler); every
# large file in each was matched by name and size against its Hub repo on
# 2026-11-08 (13 of 13 verified). The Hub holds the same files; a staging
# directory is recreated by scripts/hf_stage.sh from the trainer checkpoint.
# Explicit paths, nothing globbed. Run once:
#   bash scripts/cleanup_2026-11-08b.sh
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
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_it_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_nope_it_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_nope_qk_s1_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_nope_qk_softmax_s1_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_nope_qk_softmax_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_nope_qk_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_nope_softmax_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_nope_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_reason_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_rl_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_rope_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_sdpa_staging"
rm_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_355m_staging"
echo "removed $((total / 1024 / 1024)) GiB in total"
