#!/usr/bin/env python3
"""The HF template (triton/hf_template) against the trainer's GPT on CPU:
the modern stack (RMSNorm + SwiGLU), the uncapped learnable scale
(scale_cap 0) and the legacy path (LayerNorm + GELU, cap 15) must give the
same fp32 logits on the same weights, with strict key parity both ways.
Softmax attention only (the Stieltjes serving path is checked by the
convert job on the GPU). Also: a config with the wrong cap does NOT match,
so the cap branch is known to be wired. Then the converter's save path on
the modern model: save_pretrained with auto_map into a staging-like dir,
AutoModelForCausalLM.from_pretrained(trust_remote_code) gives the same
logits with the head still tied, and cached greedy generation equals
uncached.

    python scripts/test_hf_template_modern.py
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
sys.path.insert(0, T)
sys.path.insert(0, os.path.join(R, "triton", "dev_scripts"))
import configuration_stieltjes_gpt2 as C  # noqa: E402

src = open(os.path.join(T, "modeling_stieltjes_gpt2.py")).read().replace(
    "from .configuration_stieltjes_gpt2", "from configuration_stieltjes_gpt2")
ns = {}
exec(compile(src, "modeling_stieltjes_gpt2.py", "exec"), ns)
HF = ns["StieltjesGPT2ForCausalLM"]
from train_gpt2_stieltjes import GPT  # noqa: E402

BASE = dict(attn="sdpa", n_layer=2, n_head=2, n_embd=32, ctx=16, stj_q=4.0, rope=False,
            rope_theta=10000.0, alibi=False)


def trainer_model(**kw):
    torch.manual_seed(0)
    m = GPT(SimpleNamespace(**BASE, **kw)).eval()
    with torch.no_grad():                       # make the scale and gains matter
        for blk in m.blocks:
            if hasattr(blk.attn, "scale_mult"):
                blk.attn.scale_mult.copy_(torch.tensor([3.0, -0.8]))   # cap 15 would give 3.96, not 4
            if getattr(blk.attn, "qk_norm", False):
                blk.attn.q_gain.copy_(torch.tensor([1.3, 0.8]))
                blk.attn.k_gain.copy_(torch.tensor([0.9, 1.2]))
    return m


def hf_model(**kw):
    cfg = C.StieltjesGPT2Config(n_layer=2, n_head=2, n_embd=32, ctx=16, stj_q=4.0, attn="sdpa", **kw)
    return HF(cfg).eval()


def logits(m, ids, hf):
    with torch.no_grad():
        return m(input_ids=ids, use_cache=False).logits if hf else m(ids)[0]


def check(name, tr, hf, ids, close=True):
    sd = tr.state_dict()
    assert set(sd) == set(hf.state_dict()), (name, set(sd) ^ set(hf.state_dict()))
    hf.load_state_dict(sd, strict=True)
    a, b = logits(tr, ids, False), logits(hf, ids, True)
    ok = torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    assert ok == close, (name, (a - b).abs().max().item())
    print(f"  {name}: max |diff| {(a - b).abs().max().item():.2e} ({'match' if ok else 'differ'})")


def main():
    ids = torch.randint(0, 50304, (2, 12), generator=torch.Generator().manual_seed(1))
    modern = dict(nope=True, qk_norm=True, scale_learnable=True, scale_cap=0.0, modern=True)
    tr = trainer_model(**modern)
    assert type(tr.blocks[0].ln1).__name__ == "RMSNorm" and hasattr(tr.blocks[0].mlp, "gate")
    check("modern + qk-norm + nope + uncapped scale", tr, hf_model(**modern), ids)
    check("same weights, cap 15 in the HF config", tr, hf_model(**{**modern, "scale_cap": 15.0}), ids,
          close=False)
    legacy = dict(nope=False, qk_norm=False, scale_learnable=True, scale_cap=15.0, modern=False)
    check("legacy LayerNorm + GELU + capped scale + wpe", trainer_model(**legacy), hf_model(**legacy), ids)
    legacy_default = hf_model(nope=False, qk_norm=False, scale_learnable=True)   # no modern/scale_cap given
    assert legacy_default.config.modern is False and legacy_default.config.scale_cap == 15.0
    check("legacy through the config defaults", trainer_model(**legacy), legacy_default, ids)
    round_trip(tr, modern, ids)
    print("ALL OK")


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
        assert cj["modern"] is True and cj["scale_cap"] == 0.0 and cj["qk_norm"] is True, cj
        m2 = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True).eval()
        assert m2.head.weight.data_ptr() == m2.wte.weight.data_ptr(), "head not tied after reload"
        a, b = logits(tr, ids, False), logits(m2, ids, True)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (a - b).abs().max().item()
        with torch.no_grad():
            g = m2.generate(ids[:1, :4], max_new_tokens=5, do_sample=False)
            cur = ids[:1, :4]
            for _ in range(5):
                nxt = m2(input_ids=cur, use_cache=False).logits[0, -1].argmax()[None, None]
                cur = torch.cat([cur, nxt], 1)
        assert torch.equal(g, cur), "cached greedy differs from uncached"
        print(f"  save/AutoModel round trip: max |diff| {(a - b).abs().max().item():.2e}, "
              "head tied, cached == uncached greedy")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
