# softmax-is-meh: Stieltjes attention kernels, trainers and experiments

Code for the Stilt model family and the thesis "Stieltjes normalization as a
drop-in replacement for softmax in Transformer attention" (OSU Research
Distinction). Everything written about the results (findings, one record per
compute job, the roadmap, the paper) lives in the sibling repository
[gulkoa/softmax-research-thesis](https://github.com/gulkoa/softmax-research-thesis);
this repository holds the code, the batch scripts and the raw outputs.

## Where the results are

| resource | link |
|---|---|
| Roadmap and state of the program | https://github.com/gulkoa/softmax-research-thesis/blob/main/STILT-ROADMAP.md |
| Pretraining results report for the group (2026-10-20) | https://claude.ai/artifact/LVFkX8pNS56fFPhq8kEgVd (markdown export in the thesis repo, `findings/2026-10-20-pretraining-results-report.md`) |
| Living benchmarks doc, every model's numbers and protocols | https://claude.ai/code/artifact/540c1767-2af0-4ced-8113-f411137bfb8c (export: thesis `findings/2026-10-20-stilt-model-benchmarks-doc-export.md`) |
| Findings and experiment records | https://github.com/gulkoa/softmax-research-thesis/tree/main/findings and `/experiments` |
| Weights & Biases (all runs; consolidated 355M runs in group `355m-pretrain-consolidated`) | https://wandb.ai/gulkoa/stieltjes-flash-attn |
| Hugging Face models (`stilt.1-124m`, `stilt.1-124m-it`, `stilt.1-355m-it` public; the rest private) | https://huggingface.co/GulkoA |
| Chat demo | https://huggingface.co/spaces/GulkoA/stieltjes-gpt2-chat |
| Cluster guide (OSC Cardinal) and agent working rules | thesis repo, `workspace-docs/` |

## Layout

| path | contents |
|---|---|
| `triton/stieltjes_flash_attn.py` | the fused Stieltjes flash-attention kernel (forward and backward, Triton) |
| `triton/stieltjes.py`, `triton/fused_att.py` | standalone Stieltjes kernel and the softmax flash-attention reference |
| `triton/bench_*.py` | kernel benchmarks |
| `triton/dev_scripts/` | the GPT-2-style trainer (`train_gpt2_stieltjes.py`), evaluation scripts (long-context, passkey, bits per byte, lm-eval table) and the HF converter (`convert_verify_gpt2_hf.py`) |
| `triton/hf_template/` | the `trust_remote_code` modeling, configuration and endpoint handler that every published model ships with |
| `nanogpt/` | nanoGPT with swappable attention (early experiments) |
| `scripts/` | tooling: `test_all.sh` (the test suite, run before every commit), `hf_stage.sh` (convert and stage a checkpoint for upload), `wandb_harvest.sh` (sync offline runs), the pinned scorers (`score_stilt12.py`, `score_contrast_1e3.py`, `stability_flags.py`, `score_lr_headroom.py`, the length scorers) and the chain guards |
| `slurm/` | SLURM batch scripts; each is a versioned copy, never edited after a job used it |
| `results/` | raw outputs: job logs, evaluation outputs, `lm_eval/` results, HF staging directories, job-id lists. Checkpoints and safetensors are not committed (see `.gitignore`); they live on cluster scratch and on the Hub |
| `figures/` | figure generation scripts (run on the login node) |

## Running

- Python 3.13 via uv: the environment is `triton/.venv` (torch 2.10, triton 3.6).
- Tests: `bash scripts/test_all.sh` must pass before any commit to trainer or
  kernel code; new code gets a test in the suite.
- Cluster: OSC Cardinal H100, account `PAS2836`. Compute jobs are data-only
  (no plotting); analysis runs on the login node. See the OSC guide in the
  thesis repo's `workspace-docs/`.
- Experiment logging: every run logs to the wandb project above (offline on
  compute nodes, synced from the login node with `scripts/wandb_harvest.sh`).
- Publishing: every finished model is staged with `scripts/hf_stage.sh` and
  uploaded privately to the Hub as trained; the thesis repo records each
  publish.

## Recipe in one line

Stieltjes attention trains with normalized scores, the implicit-function-theorem
gradient and a 3-iteration Halley solver; the current base recipe adds QK-norm,
the modern backbone (RMSNorm, SwiGLU, no bias, tied head) and `torch.compile`,
at learning rate 1e-3 for 124M and 355M pilots. Details and the gotchas learned
along the way: thesis repo, `workspace-docs/agent-memory/project_stilt_insights.md`.
