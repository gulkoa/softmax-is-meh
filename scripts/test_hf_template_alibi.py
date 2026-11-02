#!/usr/bin/env python3
"""The HF template's ALiBi switch (port of 2026-11-02) against a trainer that
has --alibi, on CPU, softmax attention. The reference trainer comes from
$STILT_TRAINER (a dev_scripts dir), default the alibi worktree, which trained
the 10-14 softmax + ALiBi baseline and is the converter's reference for it
(the tok32k worktree, into which alibi was merged with bit-identical logits,
works too); main's trainer has no --alibi. The Stieltjes + ALiBi serving path
is checked against the stjalibi kernel by the convert job on the GPU.

Checks: slopes 2^(-8h/H) for h = 1..H; the softmax + ALiBi + QK-norm + modern
+ uncapped model (the baseline's flags) gives the same fp32 logits as the
trainer on the same weights, at S < ctx and at S > ctx (no position table),
with strict key parity both ways (no slopes in either state dict: the
trainer's buffer is non-persistent, the template computes them on the fly
because transformers 5.14 returns a non-persistent buffer uninitialized
from from_pretrained); the same weights with NoPE in place of ALiBi differ (the bias is
wired); the converter's save path round-trips with alibi in config.json and
no wpe, and cached greedy generation equals uncached for the softmax and the
Stieltjes ALiBi models (absolute positions in the bias under the KV cache).

    python scripts/test_hf_template_alibi.py
    STILT_TRAINER=/path/to/dev_scripts python scripts/test_hf_template_alibi.py
"""
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(R, "triton", "hf_template")
TRAINER = os.environ.get(
    "STILT_TRAINER", "/users/PAS2402/alexg/softmax/.worktrees/alibi/triton/dev_scripts")
if TRAINER.endswith(".py"):
    TRAINER = os.path.dirname(TRAINER)
sys.path.insert(0, T)
sys.path.insert(0, TRAINER)
import configuration_stieltjes_gpt2 as C  # noqa: E402

src = open(os.path.join(T, "modeling_stieltjes_gpt2.py")).read().replace(
    "from .configuration_stieltjes_gpt2", "from configuration_stieltjes_gpt2")
ns = {}
exec(compile(src, "modeling_stieltjes_gpt2.py", "exec"), ns)
HF = ns["StieltjesGPT2ForCausalLM"]
SLOPES = ns["alibi_slopes"]
from train_gpt2_stieltjes import GPT  # noqa: E402
print("  trainer:", TRAINER)

BASE = dict(attn="sdpa", n_layer=2, n_head=2, n_embd=32, ctx=16, stj_q=4.0, rope=False,
            rope_theta=10000.0, alibi=False, nope=False)
ALIBI = dict(alibi=True, nope=False, qk_norm=True, scale_learnable=True, scale_cap=0.0, modern=True)


def trainer_model(**kw):
    torch.manual_seed(0)
    m = GPT(SimpleNamespace(**{**BASE, **kw})).eval()
    with torch.no_grad():                       # make the scale and gains matter
        for blk in m.blocks:
            if hasattr(blk.attn, "scale_mult"):
                blk.attn.scale_mult.copy_(torch.tensor([3.0, -0.8]))
            if getattr(blk.attn, "qk_norm", False):
                blk.attn.q_gain.copy_(torch.tensor([1.3, 0.8]))
                blk.attn.k_gain.copy_(torch.tensor([0.9, 1.2]))
    return m


def hf_model(attn="sdpa", **kw):
    cfg = C.StieltjesGPT2Config(n_layer=2, n_head=2, n_embd=32, ctx=16, stj_q=4.0, attn=attn, **kw)
    return HF(cfg).eval()


def logits(m, ids, hf):
    with torch.no_grad():
        return m(input_ids=ids, use_cache=False).logits if hf else m(ids)[0]


def check(name, tr, hf, ids, close=True):
    sd = tr.state_dict()
    assert set(sd) == set(hf.state_dict()), (name, set(sd) ^ set(hf.state_dict()))
    assert not any("alibi" in k or "wpe" in k for k in sd), [k for k in sd if "alibi" in k or "wpe" in k]
    hf.load_state_dict(sd, strict=True)
    a, b = logits(tr, ids, False), logits(hf, ids, True)
    ok = torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    assert ok == close, (name, (a - b).abs().max().item())
    print(f"  {name}: max |diff| {(a - b).abs().max().item():.2e} ({'match' if ok else 'differ'})")


def cached_equals_uncached(name, m, ids, new=5):
    with torch.no_grad():
        g = m.generate(ids[:1, :4], max_new_tokens=new, do_sample=False)
        cur = ids[:1, :4]
        for _ in range(new):
            nxt = m(input_ids=cur, use_cache=False).logits[0, -1].argmax()[None, None]
            cur = torch.cat([cur, nxt], 1)
    assert torch.equal(g, cur), (name, g.tolist(), cur.tolist())
    print(f"  {name}: cached == uncached greedy over {new} tokens")


def round_trip(tr, kw, ids):
    """The converter's save path, then the battery's load path (AutoModel, trust_remote_code)."""
    from transformers import AutoModelForCausalLM
    hf = hf_model(**kw)
    hf.load_state_dict(tr.state_dict(), strict=True)
    hf.config.auto_map = {"AutoConfig": "configuration_stieltjes_gpt2.StieltjesGPT2Config",
                          "AutoModelForCausalLM": "modeling_stieltjes_gpt2.StieltjesGPT2ForCausalLM"}
    d = tempfile.mkdtemp()
    try:
        for f in ("modeling_stieltjes_gpt2.py", "configuration_stieltjes_gpt2.py"):
            shutil.copy(os.path.join(T, f), d)
        hf.save_pretrained(d, safe_serialization=True)
        cj = json.load(open(os.path.join(d, "config.json")))
        assert cj["alibi"] is True and cj["nope"] is False and cj["modern"] is True, cj
        m2 = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True).eval()
        assert not hasattr(m2, "wpe") and m2.blocks[0].attn.alibi is True and m2.config.alibi is True
        assert "wpe.weight" not in m2.state_dict() and not any("alibi" in k for k in m2.state_dict())
        assert m2.head.weight.data_ptr() == m2.wte.weight.data_ptr(), "head not tied after reload"
        a, b = logits(tr, ids, False), logits(m2, ids, True)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (a - b).abs().max().item()
        print(f"  save/AutoModel round trip: max |diff| {(a - b).abs().max().item():.2e}, "
              "alibi in config.json, no wpe, head tied")
        cached_equals_uncached("softmax + ALiBi after reload", m2, ids)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 50304, (2, 12), generator=g)
    long_ids = torch.randint(0, 50304, (1, 20), generator=g)          # S = 20 > ctx = 16
    tr = trainer_model(**ALIBI)
    assert hasattr(tr.blocks[0].attn, "alibi_slopes") and not hasattr(tr, "wpe")
    hf = hf_model(**ALIBI)
    assert not hasattr(hf, "wpe") and type(hf.blocks[0].ln1).__name__ == "RMSNorm" and hf.blocks[0].attn.alibi
    s = SLOPES(2)
    assert torch.allclose(s, torch.tensor([2.0 ** -4, 2.0 ** -8])), s           # H = 2
    assert torch.equal(s, tr.blocks[0].attn.alibi_slopes), (s, tr.blocks[0].attn.alibi_slopes)
    h12 = SLOPES(12)
    assert torch.allclose(h12, 2.0 ** (-8.0 * torch.arange(1, 13, dtype=torch.float32) / 12)), h12
    print("  slopes 2^(-8h/H): H = 2 -> 1/16, 1/256; H = 12 as Press et al.")
    check("softmax + ALiBi + qk-norm + modern + uncapped, S = 12", tr, hf, ids)
    check("the same at S = 20 > ctx = 16 (no position table)", tr, hf, long_ids)
    check("same weights, NoPE in place of ALiBi", tr, hf_model(**{**ALIBI, "alibi": False, "nope": True}),
          ids, close=False)
    round_trip(tr, ALIBI, ids)
    stj = hf_model(attn="stieltjes", **ALIBI)
    stj.load_state_dict(tr.state_dict(), strict=True)
    assert not torch.allclose(logits(stj, ids, True), logits(hf, ids, True)), "stieltjes == softmax?"
    cached_equals_uncached("Stieltjes + ALiBi (dense solver)", stj, ids)
    print("ALL OK")


if __name__ == "__main__":
    main()
