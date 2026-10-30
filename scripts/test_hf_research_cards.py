"""Tests for scripts/hf_research_cards.py: naming, exclusions, arms, cards."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_research_cards as H  # noqa: E402

BASE = dict(attn="stj", lr=0.002, seed=0, tag="lrsweep", qk_norm=False, scale_learnable=True,
            scale_cap=15.0, nope=True, rope=False, n_layer=12, n_embd=768, total_tokens=5e9,
            micro_bs=16, grad_accum=32, ctx=1024, stj_q=4.0)


def test_naming():
    assert H.label("ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s0.pt") == "stj-q4-nope-lrsweep-lr0.002-s0"
    assert H.repo_name("ckpt_gpt2-sdpa-nope-qkpilotls-lr0.001_s1.pt") == "stilt-research-124m-sdpa-nope-qkpilotls-lr0.001-s1"
    n = H.repo_name("ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt")
    assert n.endswith("-stilt-tok-32k-v1-s0") and len(n) <= 96


def test_final_step_and_classify():
    assert H.final_step(BASE) == 9535
    assert H.classify("ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s0.pt", BASE, 9535) == (True, "")
    ok, why = H.classify("ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s2.pt", dict(BASE, lr=1e-3, seed=2), 4500)
    assert not ok and "unfinished" in why
    tok = dict(BASE, total_tokens=4673617812.0, vocab=32768, tokenizer="stilt-tok-32k-v1", modern=True, qk_norm=True, scale_cap=0.0)
    assert H.final_step(tok) == 8913
    assert H.classify("x.pt", tok, 8913)[0]
    assert not H.classify("x.pt", dict(BASE, n_layer=24), 28609)[0]
    assert not H.classify("ckpt_gpt2-stj-q4-nope-lrsweepsmoke-lr0.002_s0.pt", dict(BASE, tag="lrsweepsmoke"), 9535)[0]
    assert "ALiBi" in H.classify("ckpt_gpt2-sdpa-alibi-qk-modern-nocap-qkpilot-lr0.001_s0.pt", dict(BASE, attn="sdpa"), 9535)[1]
    assert not H.classify("ckpt_x.fromhf.pt", BASE, 9535)[0]
    probe = dict(BASE, tag="logitprobe", lr=1e-3, seed=1, stop_step=4600, probe_every=20)
    assert H.classify("p.pt", probe, 4600) == (True, "")
    assert not H.classify("p.pt", probe, 4000)[0]


def test_trainer_and_arms():
    assert H.trainer_for(BASE) == "qknorm"
    assert H.trainer_for(dict(BASE, modern=True)) == "backbone"
    assert H.trainer_for(dict(BASE, modern=True, vocab=32768)) == "tok32k"
    assert H.arm(BASE).startswith("A:")
    assert H.arm(dict(BASE, qk_norm=True, scale_cap=0.0)).startswith("B:")
    assert H.arm(dict(BASE, attn="sdpa", scale_learnable=False)).startswith("C:")
    assert H.arm(dict(BASE, attn="sdpa", qk_norm=True, scale_learnable=False)).startswith("D:")
    assert H.arm(dict(BASE, attn="sdpa", tag="qkpilotls")).startswith("LS:")
    assert H.arm(dict(BASE, modern=True, qk_norm=True, scale_cap=0.0)).startswith("E:")
    assert H.arm(dict(BASE, attn="sdpa", modern=True, qk_norm=True, scale_cap=0.0)).startswith("F:")


def test_references():
    assert H.references(BASE) == ["findings/2026-10-17-lr-headroom-124m.md"]
    assert H.references(dict(BASE, seed=3)) == ["experiments/2026-10-18-a-2e-3-seeds-124m.md"]
    assert H.references(dict(BASE, attn="sdpa", tag="lrsweepls", lr=3e-4, seed=1)) == ["experiments/2026-10-17-lr-headroom-ls-seed1-124m.md"]
    assert H.references(dict(BASE, attn="sdpa", tag="lrsweepls", lr=1e-3, seed=2)) == ["experiments/2026-10-18-1e-3-contrast-seeds-124m.md"]
    assert H.references(dict(BASE, tag="qkpilot", lr=1e-3, seed=1)) == ["findings/2026-10-15-stability-replicates-seed1.md"]
    assert H.references(dict(BASE, tag="logitprobe"))[0] == "findings/2026-10-16-logit-growth-probe.md"


def test_card():
    c = H.card("ckpt_gpt2-stj-q4-nope-lrsweep-lr0.002_s0.pt", BASE, 9535)
    assert "# stilt-research-124m-stj-q4-nope-lrsweep-lr0.002-s0" in c
    assert "step 9,535 of 9,536 steps of 524,288 tokens (5.00B tokens)" in c
    assert "learning rate 0.002" in c and "seed 0" in c
    assert "GPT-2 BPE tokenizer" in c and "findings/2026-10-17-lr-headroom-124m.md" in c
    assert "metrics.csv" not in c
    for bad in ("overturn", "reversal", "monotone"):
        assert bad not in c
    probe = dict(BASE, attn="sdpa", scale_learnable=False, tag="logitprobe", lr=1e-3, seed=1, stop_step=4600, probe_every=20)
    pc = H.card("ckpt_gpt2-sdpa-nope-logitprobe-lr0.001_s1.pt", probe, 4600)
    assert "stopped at step 4,600 by design" in pc and "not a full training run" in pc
    tok = dict(BASE, total_tokens=4673617812.0, vocab=32768, tokenizer="stilt-tok-32k-v1", modern=True, qk_norm=True, scale_cap=0.0)
    tc = H.card("ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt", tok, 8913)
    assert "stilt-tok-32k-v1 tokenizer" in tc and "step 8,913 of 8,914 steps" in tc and "E:" in tc


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}: ok")
    print("ALL OK")
