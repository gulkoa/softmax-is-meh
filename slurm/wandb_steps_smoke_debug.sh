#!/bin/bash
#SBATCH --job-name=wandb-smoke
#SBATCH --account=PAS2836
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/users/PAS2402/alexg/softmax/softmax-is-meh/results/wandb-smoke-%j.out
#SBATCH --error=/users/PAS2402/alexg/softmax/softmax-is-meh/results/wandb-smoke-%j.err
# Smoke test for the wandb-steps branch (trainer run from the worktree):
# three chunks of a tiny stj run share ONE wandb run id — chunk a is killed
# by `timeout` (simulates SLURM TIMEOUT), b resumes to FINAL, c must no-op.
# Offline dirs land in softmax/wandb/ under project wandb-resume-test.
set -uo pipefail
cd /users/PAS2402/alexg/softmax
export WANDB_MODE=offline
export WANDB_PROJECT=wandb-resume-test
PY=/users/PAS2402/alexg/softmax/softmax-is-meh/triton/.venv/bin/python
TR=/users/PAS2402/alexg/softmax/.worktrees/wandb-steps/triton/dev_scripts/train_gpt2_stieltjes.py
MIX="web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:0.7"
MIX="${MIX},math=/fs/scratch/PAS2836/alexg/finemath_4plus:0.15"
MIX="${MIX},code=/fs/scratch/PAS2836/alexg/codeparrot_py:0.15"
ARGS="--attn stj --n-layer 2 --n-head 4 --n-embd 256 --ctx 1024 --micro-bs 16 --grad-accum 16 --total-tokens 3e8 --warmup 20 --lr 1e-3 --val-every 25 --ckpt-every 25 --data-mix ${MIX} --nope --scale-learnable --qk-norm --tag wbsmoke${SLURM_JOB_ID}"
echo "== chunk a (timeout 120s)"
SLURM_JOB_ID=${SLURM_JOB_ID}a timeout 120s $PY $TR $ARGS; echo "chunk a exit $?"
echo "== chunk b (resume to FINAL)"
SLURM_JOB_ID=${SLURM_JOB_ID}b $PY $TR $ARGS; echo "chunk b exit $?"
echo "== chunk c (FINAL no-op)"
SLURM_JOB_ID=${SLURM_JOB_ID}c $PY $TR $ARGS; echo "chunk c exit $?"
ls -d /users/PAS2402/alexg/softmax/wandb/offline-run-*wbsmoke* 2>/dev/null
echo "ALL DONE"
