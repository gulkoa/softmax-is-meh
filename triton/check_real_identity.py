"""One real training micro-step through the old (dc8dd3d) and this branch's kernel.

Loads each checkpoint with the trainer that wrote it (default: the qknorm
worktree trainer, which the LR sweep and logit probe use), draws a
micro-batch of validation tokens, and runs forward + backward under bf16
autocast once with each kernel. Then it compares the loss and every
parameter gradient bit for bit. Use checkpoints whose scores are large
(the sweep's no-norm Stieltjes runs reach max |score| ~ 1e3), a regime the
random-input identity test never reaches.

    python triton/check_real_identity.py CKPT [CKPT ...] [--trainer PATH] [--micro-bs 16]
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = "/users/PAS2402/alexg/softmax/.worktrees/qknorm/triton/dev_scripts/train_gpt2_stieltjes.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bits(t):
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--trainer", default=TRAINER)
    ap.add_argument("--micro-bs", type=int, default=16)
    a = ap.parse_args()
    T = load(a.trainer, "pilot")
    src = subprocess.run(["git", "-C", HERE, "show", "dc8dd3d:triton/stieltjes_flash_attn.py"],
                         capture_output=True, text=True, check=True).stdout
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        path_ref = os.path.join(tmp, "stj_ref_dc8dd3d.py")
        open(path_ref, "w").write(src)
        kernels = (("old", load(path_ref, "stj_ref_dc8dd3d")),
                   ("new", load(os.path.join(HERE, "stieltjes_flash_attn.py"), "stj_new")))
        for path in a.ckpts:
            ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            args = argparse.Namespace(**ck["args"])
            if args.attn != "stj":
                print(f"{os.path.basename(path)}: attn {args.attn}, skipped")
                continue
            model = T.GPT(args).cuda()
            model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
            model.train()
            x, y = T.Shards("val", args.data_mix).batch(
                a.micro_bs, args.ctx, np.random.default_rng(777), "cuda")
            res = {}
            for name, mod in kernels:
                T.stieltjes_attention = mod.stieltjes_attention
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss.backward()
                res[name] = (loss.detach().clone(),
                             {n: p.grad.detach().clone() for n, p in model.named_parameters()
                              if p.grad is not None})
            (lo, go), (ln, gn) = res["old"], res["new"]
            loss_same = torch.equal(bits(lo), bits(ln))
            diff = [n for n in go if not torch.equal(bits(go[n]), bits(gn[n]))]
            finite = all(torch.isfinite(g).all().item() for g in gn.values())
            ok = loss_same and not diff and set(go) == set(gn) and finite
            all_ok &= ok
            print(f"{os.path.basename(path)} (step {ck['step']}): loss {lo.item():.6f} / "
                  f"{ln.item():.6f}, loss bit-equal {loss_same}, {len(go) - len(diff)}/{len(go)} "
                  f"parameter grads bit-equal, finite {finite} -> {'PASS' if ok else 'FAIL'}",
                  flush=True)
            if diff:
                worst = max(diff, key=lambda n: (go[n] - gn[n]).abs().max().item())
                print(f"   differing: {diff[:6]}{' ...' if len(diff) > 6 else ''}; worst {worst} "
                      f"max|diff| {(go[worst] - gn[worst]).abs().max().item():.3e}")
            del model
            torch.cuda.empty_cache()
    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
