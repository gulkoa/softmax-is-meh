"""Test-time length-scaling probe for NoPE extrapolation (2026-07-27).

The half-trained stj-nope 355M cliffs at the 1024 train length (finding
2026-07-27). A known length-gen intervention is log-length score
scaling (temperature ~ log n) to counter attention-entropy dispersion
past train length. This tests it at EVAL time — no retraining — by
monkeypatching the attention score-scale (does NOT touch the frozen
trainer file the queued nope jobs read).

boost(L) = 1 + alpha*(log(L/train_len))  applied globally per eval
length (a good approx for the deep-tail positions, which all sit near
length L). Reports deep-tail ppl with boost off (1.0) vs on, so we see
whether scaling flattens the cliff.

Usage: python eval_longctx_scaled.py <nope_ckpt.pt> [--alpha 1.0]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt2_stieltjes as T  # noqa: E402
from train_gpt2_stieltjes import GPT  # noqa: E402
from eval_longctx_gpt2 import long_token_stream, Shards  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from types import SimpleNamespace  # noqa: E402

DEVICE = torch.device("cuda")
_BOOST = 1.0                       # set per eval length before forward

# --- monkeypatch attention score-scale (eval-only; frozen trainer untouched)
_orig_stj = T.stieltjes_attention


def _stj_boosted(*a, **kw):
    if "sm_scale" in kw:
        kw["sm_scale"] = kw["sm_scale"] * _BOOST
    return _orig_stj(*a, **kw)


_orig_sdpa = F.scaled_dot_product_attention


def _sdpa_boosted(q, k, v, **kw):
    kw.setdefault("scale", _BOOST / math.sqrt(q.shape[-1]))
    return _orig_sdpa(q, k, v, **kw)


T.stieltjes_attention = _stj_boosted
F.scaled_dot_product_attention = _sdpa_boosted


def deep_tail_ppl(model, ids, L, boost, train_len=1024, tail=512,
                  max_tokens=500_000):
    global _BOOST
    _BOOST = boost
    d0 = max(0, L - tail)
    dn = dc = 0.0
    for lo in range(0, min(len(ids), max_tokens) - L - 1, L):
        x = ids[lo:lo + L][None].to(DEVICE)
        y = ids[lo + 1:lo + L + 1][None].to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        nll = F.cross_entropy(logits[0].float(), y[0], reduction="none")
        dn += nll[d0:].sum().item()
        dc += (L - d0)
    _BOOST = 1.0
    return math.exp(dn / dc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--alpha", default="0.5,1.0,1.5,2.0",
                    help="comma-separated alphas to sweep")
    args = ap.parse_args()
    alphas = [float(a) for a in str(args.alpha).split(",")]

    tok = AutoTokenizer.from_pretrained("gpt2")
    pg = long_token_stream(tok)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    model.eval()
    train_len = cfg.ctx
    print(f"nope={getattr(cfg,'nope',False)}  alpha sweep={alphas}",
          flush=True)

    lens = (1024, 2048, 4096, 8192)
    hdr = f"{'L':>7} {'off':>10} " + " ".join(
        f"a={a:g}".rjust(10) for a in alphas)
    print(hdr, flush=True)
    for L in lens:
        off = deep_tail_ppl(model, pg, L, 1.0, train_len)
        cells = []
        for a in alphas:
            boost = 1.0 + a * math.log(max(L, train_len) / train_len)
            cells.append(deep_tail_ppl(model, pg, L, boost, train_len))
        print(f"{L:>7} {off:>10.1f} "
              + " ".join(f"{c:>10.1f}" for c in cells), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
