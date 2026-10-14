#!/bin/bash
#SBATCH --job-name=lm-eval
#SBATCH --account=PAS2836
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-eval-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm-eval-%j.err
# Zero-shot base-model battery via lm-evaluation-harness (isolated venv
# .venv-eval, datasets pre-cached on the login node; compute nodes are
# offline). Identical settings for every model, including stock GPT-2.
# Usage: sbatch lm_eval_h100.sh <model path or hub id> <tag> [extra lm_eval args]
set -euo pipefail
MODEL=${1:?model path or hub id}; TAG=${2:?tag}; shift 2
cd /users/PAS2402/alexg/softmax/softmax-is-meh/triton
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TASKS=hellaswag,piqa,arc_easy,arc_challenge,sciq,openbookqa,winogrande,boolq,lambada_openai,commonsense_qa,wikitext,mmlu
.venv-eval/bin/python -m lm_eval --model hf \
  --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=float32" \
  --tasks "$TASKS" --num_fewshot 0 --batch_size 16 --seed 1234 \
  --output_path "/users/PAS2402/alexg/softmax/softmax-is-meh/results/lm_eval/${TAG}" "$@"
echo "ALL DONE"
