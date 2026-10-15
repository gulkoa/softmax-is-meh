#!/usr/bin/env python3
"""Check that same-seed arms of train_gpt2_stieltjes start from identical weights.

Builds the model on CPU under torch.manual_seed(seed) for several arm configs
(attention mapping x fixed/learnable score scale), exactly as the trainer does
before .to(device), and compares every shared parameter bit for bit. The
learnable scale_mult starts at zero (eff = 1 + C*tanh(0) = 1), so a learnable
arm is also the same *function* as its fixed-scale twin at step 0.

    python scripts/check_init_identity.py [--trainer PATH] [--seed 1]
"""
import argparse
import importlib.util
import sys
from types import SimpleNamespace

import torch

DEFAULT = ("/users/PAS2402/alexg/softmax/.worktrees/qknorm/triton/dev_scripts/"
           "train_gpt2_stieltjes.py")


def build(mod, seed, **over):
    cfg = dict(attn="sdpa", stj_q=4.0, n_layer=12, n_head=12, n_embd=768,
               ctx=1024, nope=True, rope=False, qk_norm=False,
               scale_learnable=False, scale_cap=15.0)
    cfg.update(over)
    torch.manual_seed(seed)
    return mod.GPT(SimpleNamespace(**cfg)).state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", default=DEFAULT)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    sys.argv = [a.trainer]
    spec = importlib.util.spec_from_file_location("trainer", a.trainer)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    arms = {
        "softmax fixed (C')": dict(attn="sdpa"),
        "softmax learnable (control)": dict(attn="sdpa", scale_learnable=True),
        "Stieltjes learnable (A')": dict(attn="stj", scale_learnable=True),
    }
    sds = {k: build(mod, a.seed, **v) for k, v in arms.items()}
    ref_name = next(iter(arms))
    ref = sds[ref_name]
    ok = True
    for name, sd in sds.items():
        shared = [k for k in sd if k in ref]
        diff = [k for k in shared if not torch.equal(sd[k], ref[k])]
        extra = {k: sd[k].tolist() for k in sd if k not in ref}
        extra_vals = sorted({v for vals in extra.values() for v in vals})
        print(f"{name}: {len(shared)} shared tensors, {len(diff)} differ from "
              f"{ref_name}; extra params {len(extra)} "
              f"(values {extra_vals[:3]})")
        ok &= not diff
    print("IDENTICAL INIT" if ok else "INIT DIFFERS")


if __name__ == "__main__":
    main()
