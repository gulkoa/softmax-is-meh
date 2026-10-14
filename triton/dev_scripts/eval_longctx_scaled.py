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
_BOOST = 1.0            # global scalar boost (per eval length)
_PERPOS = None         # (alpha, train_len): per-query-position boost on q

# --- monkeypatch attention score-scale (eval-only; frozen trainer untouched)
_orig_stj = T.stieltjes_attention
_orig_sdpa = F.scaled_dot_product_attention


def _q_perpos(q):
    """Scale each query row by 1+alpha*log(ctx_len/train_len) (>=1) — a
    per-position log-n temperature, length-AGNOSTIC (no target-len
    knowledge needed; each position self-corrects for its own context
    depth). Deep-tail positions near L get the strong boost, early
    positions ~1."""
    a, tl = _PERPOS
    S = q.shape[2]
    ctx = torch.arange(1, S + 1, device=q.device, dtype=torch.float32)
    b = 1.0 + a * torch.clamp(torch.log(ctx / tl), min=0.0)
    return q * b[None, None, :, None].to(q.dtype)


def _stj_boosted(*a, **kw):
    if _PERPOS is not None and len(a) >= 1:
        a = (_q_perpos(a[0]),) + tuple(a[1:])
    elif "sm_scale" in kw:
        kw["sm_scale"] = kw["sm_scale"] * _BOOST
    return _orig_stj(*a, **kw)


def _sdpa_boosted(q, k, v, **kw):
    if _PERPOS is not None:
        q = _q_perpos(q)
    else:
        kw.setdefault("scale", _BOOST / math.sqrt(q.shape[-1]))
    return _orig_sdpa(q, k, v, **kw)


T.stieltjes_attention = _stj_boosted
F.scaled_dot_product_attention = _sdpa_boosted


_TOK = None            # the model's tokenizer (for bytes of the scored tail)
_TAILB = {}            # (L, lo) -> utf-8 bytes of the tail targets
_MAXTOK = 500_000      # tokens of the stream to score (all of it in --text-chars mode)


def _tail_bytes(ids, key):
    if key not in _TAILB:
        _TAILB[key] = len(_TOK.decode(ids, skip_special_tokens=False).encode("utf-8"))
    return _TAILB[key]


def deep_tail_ppl(model, ids, L, boost, train_len=1024, tail=512,
                  max_tokens=None, want_bpb=False):
    """Deep-tail token perplexity (and, with want_bpb, bits per byte of the
    same tail — the tokenizer-independent number)."""
    global _BOOST
    _BOOST = boost
    max_tokens = max_tokens or _MAXTOK
    d0 = max(0, L - tail)
    dn = dc = db = 0.0
    for lo in range(0, min(len(ids), max_tokens) - L - 1, L):
        x = ids[lo:lo + L][None].to(DEVICE)
        y = ids[lo + 1:lo + L + 1][None].to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        nll = F.cross_entropy(logits[0].float(), y[0], reduction="none")
        dn += nll[d0:].sum().item()
        dc += (L - d0)
        if want_bpb:
            db += _tail_bytes(y[0][d0:].tolist(), (L, lo))
    _BOOST = 1.0
    ppl = math.exp(dn / dc)
    return (ppl, dn / (math.log(2) * db)) if want_bpb else ppl


def pg_text_stream(tok, n_chars):
    """The first n_chars characters of the PG-19 test books, tokenized with
    the model's tokenizer — the SAME TEXT under every tokenizer."""
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    text = ""
    for ex in ds:
        text += ex["text"] + "\n\n"
        if len(text) >= n_chars:
            break
    text = text[:n_chars]
    ids = tok(text, add_special_tokens=False).input_ids
    print(f"text stream: {n_chars} chars, {len(text.encode('utf-8'))} bytes -> {len(ids)} tokens "
          f"({len(text.encode('utf-8'))/len(ids):.3f} B/tok)", flush=True)
    return torch.tensor(ids, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--alpha", default="0.5,1.0,1.5,2.0",
                    help="comma-separated alphas to sweep")
    ap.add_argument("--per-position", default=None, dest="perpos",
                    help="alpha for length-agnostic per-position boost "
                         "(one setting, all lengths)")
    ap.add_argument("--text-chars", type=int, default=0, dest="text_chars",
                    help="build the PG-19 stream from the first N characters (same text "
                         "under every tokenizer) and report bits per byte beside ppl")
    ap.add_argument("--theta-mult", default=None, dest="theta_mult",
                    help="RoPE models: comma-separated rope_theta "
                         "multipliers to sweep (NTK-style context "
                         "extension); replaces the alpha sweep")
    args = ap.parse_args()
    alphas = [float(a) for a in str(args.alpha).split(",")]

    _args = torch.load(args.ckpt, map_location="cpu", weights_only=False)["args"]
    _tp = _args.get("tokenizer_path") if _args.get("tokenizer", "gpt2") != "gpt2" else None
    tok = AutoTokenizer.from_pretrained(_tp or "gpt2")
    print(f"tokenizer: {_args.get('tokenizer', 'gpt2')} ({_tp or 'hub gpt2'})", flush=True)
    global _TOK, _MAXTOK
    _TOK = tok
    if args.text_chars:
        pg = pg_text_stream(tok, args.text_chars)
        _MAXTOK = len(pg)
    else:
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
    if args.theta_mult is not None:
        assert getattr(cfg, "rope", False), "--theta-mult needs a rope ckpt"
        mults = [float(m) for m in str(args.theta_mult).split(",")]
        base = float(getattr(cfg, "rope_theta", 10000.0))
        hdr = f"{'L':>7} " + " ".join(f"th x{m:g}".rjust(10) for m in mults)
        print(hdr, flush=True)
        attns = [m for m in model.modules() if hasattr(m, "_rope_cache")]
        # optional combined grid: sharpening boost on top of theta scaling
        combo_alphas = ([0.0] if args.alpha == ap.get_default("alpha")
                        else [float(x) for x in str(args.alpha).split(",")])
        for L in lens:
            cells = []
            for mult in mults:
                cfg.rope_theta = base * mult
                for a_ in attns:
                    a_._rope_cache = None
                best = None
                for ca in combo_alphas:
                    boost = 1.0 + ca * math.log(max(L, train_len) / train_len)
                    v = deep_tail_ppl(model, pg, L, boost, train_len)
                    best = v if best is None or v < best else best
                cells.append(best)
            cfg.rope_theta = base
            for a_ in attns:
                a_._rope_cache = None
            print(f"{L:>7} " + " ".join(f"{c:>10.1f}" for c in cells),
                  flush=True)
        print("DONE", flush=True)
        return
    global _PERPOS
    if args.perpos is not None:
        a = float(args.perpos)
        print(f"per-position (length-agnostic) alpha={a}", flush=True)
        print(f"{'L':>7} {'off':>10} {'perpos':>10}", flush=True)
        for L in lens:
            _PERPOS = None
            off = deep_tail_ppl(model, pg, L, 1.0, train_len)
            _PERPOS = (a, train_len)
            on = deep_tail_ppl(model, pg, L, 1.0, train_len)
            _PERPOS = None
            print(f"{L:>7} {off:>10.1f} {on:>10.1f}", flush=True)
        print("DONE", flush=True)
        return
    hdr = f"{'L':>7} {'off':>10} " + " ".join(
        f"a={a:g}".rjust(10) for a in alphas)
    print(hdr, flush=True)
    bpb_rows = []
    for L in lens:
        wb = bool(args.text_chars)
        off = deep_tail_ppl(model, pg, L, 1.0, train_len, want_bpb=wb)
        cells = []
        for a in alphas:
            boost = 1.0 + a * math.log(max(L, train_len) / train_len)
            cells.append(deep_tail_ppl(model, pg, L, boost, train_len, want_bpb=wb))
        if wb:
            bpb_rows.append((L, off[1], [c[1] for c in cells]))
            off, cells = off[0], [c[0] for c in cells]
        print(f"{L:>7} {off:>10.1f} "
              + " ".join(f"{c:>10.1f}" for c in cells), flush=True)
    if bpb_rows:
        print("deep-tail bits per byte (same text under every tokenizer):", flush=True)
        print(hdr, flush=True)
        for L, off, cells in bpb_rows:
            print(f"{L:>7} {off:>10.4f} " + " ".join(f"{c:>10.4f}" for c in cells), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
