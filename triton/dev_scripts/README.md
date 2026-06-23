# dev_scripts — Stieltjes flash-attention development & diagnostic scripts

Archive of the development/diagnostic/test scripts for the Triton Stieltjes
kernel work (2026-05 / 2026-06). These run on a GPU via the matching SLURM
scripts in `../../slurm/` (which reference them by their original `tmp/...`
path; the copies here are the version-controlled record).

Key scripts:

- `test_triton_normalized.py` — correctness suite for `normalize=True` mode
  (forward/backward vs PyTorch normalized `stieltjes`, NaN sweep).
- `flashnorm_archnew.py` — end-to-end: full normalized kernel in Jack's
  `architecture_new`, vs the PyTorch reference.
- `why_norm_helps.py` — 4-variant factorization isolating WHY normalization
  helps OOD (it is the backward gradient).
- `grad_structure_analysis.py` / `verify_normalized_grad.py` — analytic
  normalized-Stieltjes gradient, verified vs autograd to machine precision.
- `flashfwd_torchbwd_archnew.py` / `flashfull_archnew.py` — step-1/step-3
  validation (flash fwd + torch bwd; full flash fwd+bwd) in `architecture_new`.
- `maxretr_bs_vs_nr.py`, `bs_fwd_ift_bwd.py`, `nrfwd_bsbwd.py` — bisection vs
  Newton-Raphson gradient-flow study (the backward drives the OOD gap).
- `nan_truth_table_gpu.py`, `nan_mechanism_cpu.py`, `quantify_padded_keycol_bug.py`,
  `verify_fix_selftests.py`, `retest_and_localize.py` — NaN root-cause + the
  padded-key-column forward bug (BUG 2) investigation and fixes.
- `diagnose_bsmode_q4_error.py`, `check_triton_block_lambda_grad.py` — BS-mode
  argmax/near-tie diagnostics.

Findings written up in `thesis/findings/` (2026-05-12, 2026-05-26, 2026-06-09).
Kernel modes: default (IFT), `block_lambda_grad=True` (unnormalized BS,
matches `stieltjes_old`), `normalize=True` (matches `stieltjes`).
