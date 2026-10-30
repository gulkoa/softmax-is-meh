"""Pick the finished 124M research checkpoints, name their Hub repos, and
write a model card into a staging directory for each (policy of
2026-10-30: every finished training checkpoint goes to the Hub, privately,
under the prefix stilt-research-124m-<run label>; see
thesis/experiments/2026-10-30-hf-upload-research-checkpoints-124m.md).

Writes LIST lines "<ckpt> <staging> <trainer dev_scripts dir> <repo>" for
slurm/convert_many_hf_h100.sh and scripts/hf_upload_verified.sh, and
README.md into each staging dir. Excluded, with the reason printed: 355M
checkpoints (published separately), smoke runs, ALiBi runs (no template
switch yet), probe artifacts, unfinished runs (uploaded at FINAL).

    python scripts/hf_research_cards.py [--scan DIR] [--staging-root DIR]
                                        [--list FILE] [--dry]
"""
import argparse
import glob
import os
import sys

SCAN = "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt"
STAGING_ROOT = "/fs/scratch/PAS2836/alexg/hf_research_staging"
LIST = "/users/PAS2402/alexg/softmax/softmax-is-meh/results/hf_research_list.txt"
WT = "/users/PAS2402/alexg/softmax/.worktrees"
TRAINERS = {"qknorm": f"{WT}/qknorm/triton/dev_scripts",
            "backbone": f"{WT}/backbone/triton/dev_scripts",
            "tok32k": f"{WT}/tok32k/triton/dev_scripts"}
PREFIX = "stilt-research-124m-"
THESIS = "https://github.com/gulkoa/softmax-research-thesis"


def label(basename):
    """ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s0.pt -> stj-q4-nope-lrsweep-lr0.002-s0"""
    b = basename
    for pre in ("ckpt_gpt2-",):
        if b.startswith(pre):
            b = b[len(pre):]
    if b.endswith(".pt"):
        b = b[:-3]
    return b.replace("_s", "-s")


def repo_name(basename):
    return PREFIX + label(basename)


def trainer_for(args):
    if "vocab" in args:
        return "tok32k"
    if args.get("modern"):
        return "backbone"
    return "qknorm"


def final_step(args):
    tps = int(args.get("micro_bs", 0)) * int(args.get("grad_accum", 0)) * int(args.get("ctx", 0)) or 524288
    return int(float(args["total_tokens"]) // tps) - 1


def classify(basename, args, step):
    """(include, reason). reason explains an exclusion."""
    if int(args.get("n_layer", 12)) != 12:
        return False, "355M checkpoint, published separately"
    if "smoke" in args.get("tag", "") or "smoke" in basename:
        return False, "smoke run"
    if args.get("alibi") or "alibi" in basename:
        return False, "ALiBi run: the HF template has no ALiBi switch yet"
    if "fromhf" in basename:
        return False, "round-trip probe artifact"
    if str(args.get("tag", "")).startswith("logitprobe"):
        if args.get("stop_step") and int(step) == int(args["stop_step"]):
            return True, ""
        return False, f"probe run not at its stop step ({step} vs {args.get('stop_step')})"
    fs = final_step(args)
    if int(step) < fs:
        return False, f"unfinished (step {step} of {fs})"
    return True, ""


def arm(args):
    stj = args["attn"] == "stj"
    qk = bool(args.get("qk_norm"))
    sl = bool(args.get("scale_learnable"))
    cap = float(args.get("scale_cap", 15.0))
    modern = bool(args.get("modern"))
    if stj and qk and modern:
        return "E: Stieltjes attention on the modern backbone (NoPE, QK-norm, uncapped learnable score scale)"
    if not stj and qk and modern:
        return "F: softmax attention on the modern backbone (NoPE, QK-norm, uncapped learnable score scale)"
    if stj and qk:
        return "B: Stieltjes attention with QK-norm and an uncapped learnable score scale"
    if stj:
        return f"A: Stieltjes attention with a learnable score scale (tanh cap {cap:g}) and no normalization"
    if qk:
        return "D: softmax attention with QK-norm and a fixed score scale"
    if sl:
        return f"LS: softmax attention with the learnable capped score scale (tanh cap {cap:g}) and no normalization, the scale control"
    return "C: softmax attention with a fixed score scale and no normalization"


def references(args):
    """Record or finding files behind this run, by campaign."""
    tag = str(args.get("tag", ""))
    lr = float(args["lr"])
    seed = int(args.get("seed", 0))
    if tag.startswith("logitprobe"):
        return ["findings/2026-10-16-logit-growth-probe.md", "experiments/2026-10-15-logit-growth-probe.md"]
    if "vocab" in args:
        return ["findings/2026-10-14-plan-tokenizer-32k.md", "findings/2026-10-14-plan-stilt12-355m-pair.md"]
    if args.get("modern"):
        return (["findings/2026-08-30-modern-backbone-verdict.md"] if seed == 0
                else ["experiments/2026-10-18-ef-seed1-124m.md", "findings/2026-10-17-length-lead-alpha-grid.md"])
    if tag == "qkpilotls":
        return ["findings/2026-10-15-scale-control-caveat-removed.md"]
    if tag == "qkpilot":
        if seed == 0:
            return ["findings/2026-08-27-qknorm-gate-verdict.md"]
        return ["findings/2026-10-15-stability-replicates-seed1.md"]
    if tag in ("lrsweep", "lrsweepls"):
        if abs(lr - 1e-3) < 1e-9:
            return ["experiments/2026-10-18-1e-3-contrast-seeds-124m.md"]
        if seed == 0:
            return ["findings/2026-10-17-lr-headroom-124m.md"]
        if tag == "lrsweep":
            return (["experiments/2026-10-16-lr-headroom-a-seed1-124m.md"] if seed == 1
                    else ["experiments/2026-10-18-a-2e-3-seeds-124m.md"])
        return (["experiments/2026-10-17-lr-headroom-ls-seed1-124m.md"] if seed == 1
                else ["experiments/2026-10-18-ls-3e-4-seeds-124m.md"])
    return []


def card(basename, args, step):
    name = repo_name(basename)
    stj = args["attn"] == "stj"
    tok = args.get("tokenizer", "gpt2") or "gpt2"
    tok_s = ("the stilt-tok-32k-v1 tokenizer (32,768 tokens)" if tok != "gpt2"
             else "the GPT-2 BPE tokenizer (50,257 tokens)")
    tokens = float(args["total_tokens"])
    tps = int(args.get("micro_bs", 0)) * int(args.get("grad_accum", 0)) * int(args.get("ctx", 0)) or 524288
    probe = str(args.get("tag", "")).startswith("logitprobe")
    if probe:
        run_s = (f"The run was stopped at step {int(step):,} by design: it is the logit-growth probe, "
                 f"which logged the largest attention score per layer every "
                 f"{int(args.get('probe_every', 20))} steps, so this checkpoint is not a full training run.")
    else:
        run_s = (f"This is the final checkpoint, step {int(step):,} of "
                 f"{int(tokens // tps):,} steps of {tps:,} tokens ({tokens / 1e9:.2f}B tokens).")
    backbone = ("modern backbone (RMSNorm, SwiGLU, no bias, tied head)" if args.get("modern")
                else "GPT-2 backbone (LayerNorm, GELU)")
    scale = ("off" if not args.get("scale_learnable")
             else ("on, uncapped" if float(args.get("scale_cap", 15.0)) == 0
                   else f"on, tanh cap {float(args.get('scale_cap', 15.0)):g}"))
    refs = "\n".join(f"- `{r}`" for r in references(args)) or "- see the roadmap"
    return f"""---
license: mit
tags:
  - text-generation
  - research
  - custom_code
library_name: transformers
pipeline_tag: text-generation
---

# {name}

A research checkpoint from the Stilt program (Stieltjes attention as a
drop-in replacement for softmax), not a Stilt release. 124M parameters,
GPT-2-small shape (12 layers, 12 heads, width 768), context 1024,
{tok_s}. Trained on a 70/15/15 mix of FineWeb-Edu, FineMath and Python
code with seed {int(args.get('seed', 0))} at learning rate {float(args['lr']):g}.

Configuration: {arm(args)}. Attention {'Stieltjes (q = 4)' if stj else 'softmax'};
positions: none (NoPE); QK-norm {'on' if args.get('qk_norm') else 'off'};
learnable score scale {scale}; {backbone}.

{run_s} The record and the reading it belongs to, in
[the thesis repository]({THESIS}):

{refs}

Base (completion-style) model, no instruction tuning. Numbers and
verdicts live in those files, not on this card.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "GulkoA/{name}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

ids = tok("The history of mathematics begins with", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=40, do_sample=True, top_k=40)
print(tok.decode(out[0]))
```
"""


def load_args(path):
    import torch
    blob = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return dict(blob["args"]), int(blob.get("step", -1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=SCAN)
    ap.add_argument("--staging-root", default=STAGING_ROOT)
    ap.add_argument("--list", default=LIST)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    paths = sorted(p for p in glob.glob(os.path.join(a.scan, "ckpt_*.pt"))
                   if not any(x in os.path.basename(p) for x in (".step", ".bak", ".corrupt")))
    lines, excluded = [], []
    for p in paths:
        b = os.path.basename(p)
        args, step = load_args(p)
        ok, why = classify(b, args, step)
        if not ok:
            excluded.append((b, why))
            continue
        staging = os.path.join(a.staging_root, label(b))
        lines.append(f"{p} {staging} {TRAINERS[trainer_for(args)]} GulkoA/{repo_name(b)}")
        if not a.dry:
            os.makedirs(staging, exist_ok=True)
            with open(os.path.join(staging, "README.md"), "w") as f:
                f.write(card(b, args, step))
    if not a.dry:
        with open(a.list, "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} checkpoints listed, {len(excluded)} excluded")
    for b, why in excluded:
        print(f"  excluded {b}: {why}")
    for ln in lines:
        print("  " + ln.split()[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
