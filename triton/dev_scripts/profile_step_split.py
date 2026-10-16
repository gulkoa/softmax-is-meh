"""Where does a training step's GPU time go, per attention mapping?

Builds the main trainer's GPT (train_gpt2_stieltjes.py, imported read-only;
its main() is not run) at a given shape, default the 355M P2 seed-1 config
(24 x 16 x 1024, ctx 1,024, micro-batch 8, NoPE, QK-norm, capped learnable
scale, bf16 autocast, q = 4, Halley-3, normalize, IFT gradient), and for each
mapping reports:
  1. wall time per micro-step (forward + backward, CUDA events), and per
     optimizer step (clip + AdamW, CUDA events);
  2. kernel time per micro-step split into attention forward, attention
     backward and everything else (torch.profiler), plus the largest
     "everything else" kernels;
  3. at the same attention shape, the Stieltjes forward alone at
     num_iter = 1, 2, 3 (the marginal cost of one Halley sweep) and its
     backward, against SDPA's forward and backward.
Measurement only. The data loader and the trainer's logging are not included.

    python triton/dev_scripts/profile_step_split.py [--layers 24 --heads 16
        --embd 1024 --micro-bs 8] [--iters 8] [--json OUT]
    python triton/dev_scripts/profile_step_split.py --dry   # CPU plumbing check
    --kernel PATH swaps in another stieltjes_flash_attn.py (e.g. a branch's)
    for the trainer's, in the model and in the attention-only timings.
"""
import argparse
import importlib.util
import json
import sys
import types
from collections import defaultdict

import torch

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton/dev_scripts")
import train_gpt2_stieltjes as T  # noqa: E402  (module import only)


def category(name):
    """Kernel name -> attn_fwd / attn_bwd / rest."""
    n = name.lower()
    if "_stieltjes_attn_fwd" in n:
        return "attn_fwd"
    if "_stieltjes_bwd" in n:
        return "attn_bwd"
    if "flash" in n or "fmha" in n:
        return "attn_bwd" if ("bwd" in n or "backward" in n or "cutlassb" in n) else "attn_fwd"
    return "rest"


def make_model(attn, a, device):
    cfg = types.SimpleNamespace(
        attn=attn, stj_q=4.0, n_layer=a.layers, n_head=a.heads, n_embd=a.embd,
        ctx=a.ctx, nope=True, rope=False, scale_learnable=True, scale_cap=15.0,
        qk_norm=True, modern=False)
    torch.manual_seed(0)
    model = T.GPT(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=1e-4, betas=(0.9, 0.95))
    return model, opt


def step_split(attn, a, device):
    model, opt = make_model(attn, a, device)
    x = torch.randint(0, T.VOCAB, (a.micro_bs, a.ctx + 1), device=device)
    inp, tgt = x[:, :-1], x[:, 1:]
    amp = device == "cuda"

    def micro():
        with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
            _, loss = model(inp, tgt)
        loss.backward()

    def opt_step():
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=False)

    for _ in range(a.warmup):
        micro()
    opt_step()
    if not amp:
        return {"attn": attn, "dry": True}
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(a.iters):
        micro()
    e1.record()
    torch.cuda.synchronize()
    micro_ms = e0.elapsed_time(e1) / a.iters
    e0.record()
    for _ in range(3):
        opt_step()
    e1.record()
    torch.cuda.synchronize()
    opt_ms = e0.elapsed_time(e1) / 3

    acts = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=acts) as prof:
        for _ in range(a.iters):
            micro()
        torch.cuda.synchronize()
    cat, per_kernel = defaultdict(float), defaultdict(float)
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        us = ev.time_range.elapsed_us()
        cat[category(ev.name)] += us
        per_kernel[ev.name] += us
    split = {k: cat[k] / 1000 / a.iters for k in ("attn_fwd", "attn_bwd", "rest")}
    top = sorted(((v / 1000 / a.iters, k) for k, v in per_kernel.items()
                  if category(k) == "rest"), reverse=True)[:8]
    return {"attn": attn, "micro_ms": micro_ms, "opt_ms": opt_ms,
            "kernel_ms": split, "top_rest": [(round(ms, 3), k[:90]) for ms, k in top],
            "tok_per_s_gpu_only": a.micro_bs * a.ctx / (micro_ms / 1000
                                                        + opt_ms / 1000 / a.grad_accum)}


def time_ms(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def attn_micro(a):
    """Attention alone at the trainer's shape (B = micro-batch, H, N, D)."""
    import math
    B, H, N, D = a.micro_bs, a.heads, a.ctx, a.embd // a.heads
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16,
                           generator=g).requires_grad_() for _ in range(3))
    do = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16, generator=g)
    sm = 1.0 / math.sqrt(D)
    out = {}

    def stj(it):
        return T.stieltjes_attention(q, k, v, causal=True, sm_scale=sm, stieltjes_q=4.0,
                                     num_iter=it, solver="halley", normalize=True,
                                     ift_grad=True)

    with torch.no_grad():
        for it in (1, 2, 3):
            out[f"stj_fwd_iter{it}_ms"] = time_ms(lambda: stj(it), a.iters * 4)
        out["sdpa_fwd_ms"] = time_ms(lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True), a.iters * 4)

    def fb(fn):
        def run():
            o = fn()
            o.backward(do)
            q.grad = k.grad = v.grad = None
        return run

    out["stj_fwdbwd_iter3_ms"] = time_ms(fb(lambda: stj(3)), a.iters * 2)
    out["sdpa_fwdbwd_ms"] = time_ms(fb(lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True)), a.iters * 2)
    out["stj_bwd_iter3_ms"] = out["stj_fwdbwd_iter3_ms"] - out["stj_fwd_iter3_ms"]
    out["sdpa_bwd_ms"] = out["sdpa_fwdbwd_ms"] - out["sdpa_fwd_ms"]
    out["shape"] = [B, H, N, D]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--embd", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--json", default=None)
    ap.add_argument("--kernel", default=None,
                    help="stieltjes_flash_attn.py to use instead of the trainer's")
    ap.add_argument("--dry", action="store_true",
                    help="CPU plumbing check: tiny model, sdpa only, no timing")
    a = ap.parse_args()
    if a.kernel:
        spec = importlib.util.spec_from_file_location("stj_kernel_override", a.kernel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        T.stieltjes_attention = mod.stieltjes_attention   # Attn.forward looks it up here
        print("kernel override:", a.kernel, flush=True)
    if a.dry:
        a.layers, a.heads, a.embd, a.ctx, a.micro_bs, a.warmup = 2, 2, 64, 32, 2, 1
        assert category("_stieltjes_attn_fwd") == "attn_fwd"
        assert category("_stieltjes_bwd_dkdv") == "attn_bwd"
        assert category("void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits>") == "attn_fwd"
        assert category("void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel") == "attn_bwd"
        assert category("fmha_cutlassB_bf16_aligned_64x64_k64_sm80") == "attn_bwd"
        assert category("sm90_xmma_gemm_bf16bf16_bf16f32") == "rest"
        print(step_split("sdpa", a, "cpu"))
        print("dry ok")
        return
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}; shape "
          f"{a.layers}x{a.heads}x{a.embd}, ctx {a.ctx}, micro-batch {a.micro_bs}", flush=True)
    res = {"args": vars(a), "steps": {}, "attention": None}
    for attn in ("sdpa", "stj"):
        r = step_split(attn, a, "cuda")
        res["steps"][attn] = r
        ks = r["kernel_ms"]
        tot = sum(ks.values())
        print(f"\n[{attn}] micro-step {r['micro_ms']:.2f} ms wall, optimizer step "
              f"{r['opt_ms']:.2f} ms; kernel time {tot:.2f} ms = attn fwd "
              f"{ks['attn_fwd']:.2f} ({ks['attn_fwd'] / tot:.1%}) + attn bwd "
              f"{ks['attn_bwd']:.2f} ({ks['attn_bwd'] / tot:.1%}) + rest {ks['rest']:.2f} "
              f"({ks['rest'] / tot:.1%}); GPU-only {r['tok_per_s_gpu_only'] / 1000:.0f}k tok/s",
              flush=True)
        for ms, k in r["top_rest"]:
            print(f"    {ms:8.3f} ms  {k}")
        torch.cuda.empty_cache()
    s, t = res["steps"]["sdpa"], res["steps"]["stj"]
    gap = t["micro_ms"] - s["micro_ms"]
    attn_gap = (sum(t["kernel_ms"][k] for k in ("attn_fwd", "attn_bwd"))
                - sum(s["kernel_ms"][k] for k in ("attn_fwd", "attn_bwd")))
    rest_gap = t["kernel_ms"]["rest"] - s["kernel_ms"]["rest"]
    print(f"\nstj / sdpa micro-step wall: {t['micro_ms'] / s['micro_ms']:.3f} "
          f"(gap {gap:.2f} ms); attention kernel gap {attn_gap:.2f} ms, rest-kernel gap "
          f"{rest_gap:.2f} ms. Bound: Stieltjes attention as fast as SDPA's would give "
          f"{t['micro_ms'] - attn_gap:.2f} ms ({(t['micro_ms'] - attn_gap) / s['micro_ms']:.3f}x "
          f"sdpa).", flush=True)
    res["attention"] = am = attn_micro(a)
    print("\nattention alone at shape", am["shape"])
    for k, v in am.items():
        if k != "shape":
            print(f"    {k:24s} {v:8.3f} ms")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
