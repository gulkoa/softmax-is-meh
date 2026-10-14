#!/bin/bash
# Zero-shot battery (lm-evaluation-harness) on every pretrained base + stock
# GPT-2, identical settings. Models from the local HF staging dirs (= the Hub
# files); stock GPT-2 from the HF cache (canonical ids, cached 10-14).
#   bash slurm/submit_lm_eval_20261014.sh
set -euo pipefail
S=/users/PAS2402/alexg/softmax/softmax-is-meh/slurm/lm_eval_h100.sh
R=/users/PAS2402/alexg/softmax/softmax-is-meh/results
while read -r model tag; do
  j=$(sbatch --parsable "$S" "$model" "$tag"); echo "$j  $tag"
done <<LIST
$R/hf_355m_staging stilt.1-355m
$R/hf_355m_sdpa_staging stilt.1-355m-softmax-baseline
$R/hf_355m_nope_staging stilt.1.1-355m-nope
$R/hf_355m_nope_softmax_staging stilt.1.1-355m-nope-softmax-baseline
$R/hf_355m_rope_staging stilt.1.1-355m-rope
$R/hf_355m_nope_qk_staging stilt.1.1-355m-nope-qk
$R/hf_355m_nope_qk_softmax_staging stilt.1.1-355m-nope-qk-softmax-baseline
$R/hf_gpt2_staging stilt.1-124m
openai-community/gpt2-medium gpt2-medium
openai-community/gpt2 gpt2
LIST
