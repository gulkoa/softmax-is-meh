#!/usr/bin/env python3
"""Tests for scripts/score_stilt12.py (CPU, synthetic logs; one test also
reads the committed 124M uniform-grid length log when it is present).

    python scripts/test_score_stilt12.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_stilt12 as sc  # noqa: E402
from score_stilt12 import (NEW, OLD, S11, S12, T11, T12, TAGS, chunk_logs, parse_bpb,  # noqa: E402
                           reading1, reading2, reading3, stability, versus_11)

D = tempfile.mkdtemp()


def write(name, text):
    p = os.path.join(D, name)
    open(p, "w").write(text)
    return p


def train_log(vals, final=False, gnorm=0.5):
    """A chunk log: 20-step train lines from the first to the last validation step, then [val] lines."""
    lo, hi = min(vals), max(vals)
    lines = [f"step {s:6d}/26742 loss 3.0000 gnorm {gnorm:.2f} (80k tok/s)"
             for s in range(lo - lo % 20, hi + 1, 20)]
    lines += [f"  [val] step {s} loss {p:.4f} ppl {p:.2f} (wall 1.00h, 1.00B tok, 1M tok/GPU-h)"
              for s, p in sorted(vals.items())]
    if final:
        lines.append("FINAL ckpt: /fs/scratch/x.pt")
    return "\n".join(lines) + "\n"


def smooth(lo=250, hi=26500, start=40.0):
    """A falling validation curve every 250 steps."""
    return {s: round(start * (1 - 0.6 * s / hi), 2) for s in range(lo, hi + 1, 250)}


def test_chunk_logs_skip_smoke_and_keep_order():
    j = write("jobs.txt", "smoke (stj, 355M, compile): 100\nstj chunk 1: 101\nstj chunk 2: 102\n"
                          "sdpa chunk 1: 201\nstj chunk 3: 103\n")
    arms = chunk_logs(j, "/r")
    assert arms == {"stj": ["/r/stilt12-101.out", "/r/stilt12-102.out", "/r/stilt12-103.out"],
                    "sdpa": ["/r/stilt12-201.out"]}


def test_stability_incomplete_prints_no_curve_reading():
    v = smooth(hi=20000)
    v[10000] = 99.0                                   # a jump, but the run is not complete
    p = write("inc.out", train_log(v))
    lines, mono = stability([p, os.path.join(D, "never-started.out")])
    assert mono is None and len(lines) == 1
    assert lines[0] == "incomplete (last validation at step 20000); not scored"
    v = smooth()
    p = write("nofinal.out", train_log(v))            # val at 26,500 but no FINAL line
    assert stability([p])[1] is None
    assert stability([os.path.join(D, "nothing.out")]) == (["no chunk log yet; not scored"], None)


def test_stability_monotone_jump_and_rules():
    v = smooth()
    a = write("a.out", train_log({s: x for s, x in v.items() if s <= 13000}))
    b = write("b.out", train_log({s: x for s, x in v.items() if s > 13000}, final=True))
    lines, mono = stability([a, b])
    assert mono is True and lines[-1].startswith("monotone at 1e-3")
    assert "validation perplexity" in lines[0] and "at step 26500" in lines[0]
    # a jump above 1.05x after step 2,000 fails
    j = dict(v)
    j[5000] = round(j[4750] * 1.06, 2)
    lines, mono = stability([write("j.out", train_log(j, final=True))])
    assert mono is False and "first at step 5000" in lines[-1] and "1 validation jump" in lines[-1]
    # exactly 1.05x is not a jump; a jump at step 2,000 or before is outside the rule
    k = dict(v)
    k[2000] = round(k[1750] * 1.5, 2)
    k[26250], k[26500] = 10.00, 10.50
    assert stability([write("k.out", train_log(k, final=True))])[1] is True
    # a non-finite validation after step 2,000 fails
    n = dict(v)
    n[8000] = float("nan")
    assert stability([write("n.out", train_log(n, final=True))])[1] is False
    # gradient norms above 5 are descriptive only
    lines, mono = stability([write("g.out", train_log(v, final=True, gnorm=7.0))])
    assert mono is True and any(x.startswith("descriptive: ") and "above 5" in x and "largest 7.00" in x
                                for x in lines)


def test_stability_later_chunk_wins():
    v = smooth()
    bad = {s: x for s, x in v.items() if s <= 6000}
    bad[6000] = round(bad[5750] * 1.2, 2)             # spike logged, then the steps re-run
    a = write("w1.out", train_log(bad))
    b = write("w2.out", train_log({s: x for s, x in v.items() if s >= 5750}, final=True))
    assert stability([a, b])[1] is True
    assert stability([b, a])[1] is False              # order matters: submission order


def test_reading2_sentences():
    assert reading2({"stj": True, "sdpa": True}) == ["reading 2: both arms monotone at 1e-3"]
    out = reading2({"stj": True, "sdpa": False})
    assert out == ["reading 2: a softmax spike at 1e-3 despite QK-norm, a new result (as pre-registered)"]
    out = reading2({"stj": False, "sdpa": None})
    assert out[0].startswith("reading 2: a Stieltjes spike at 1e-3, which retracts")
    assert out[-1] == "reading 2: incomplete until both runs are complete"
    assert reading2({"stj": None, "sdpa": None}) == ["reading 2: incomplete until both runs are complete"]


def bpb_block(ck, mix, bpb, docs=1500, evaluated=None):
    return (f"=== {ck} | {mix}\n"
            f"ckpt {evaluated or ck} | tokenizer stilt-tok-32k-v1 (EOT 0) | ctx 1024 | {docs} docs per source\n"
            "source    tokens      bytes  B/tok  token ppl     BPB\n"
            f"web      1498112    6806592  4.543     23.572  {bpb:.4f}\n"
            f"mix-weighted BPB (70/15/15): {bpb:.4f}\n")


def lm_results(acc, wt=0.9, lam=None):
    r = {k: {"acc,none": acc} for k in sc.NINE}
    r["lambada_openai"] = {"acc,none": acc if lam is None else lam}
    r["wikitext"] = {"bits_per_byte,none": wt}
    return r


def test_parse_bpb_and_reading1():
    log = write("bpb.out", "S12 step 26741 final step 26741\n" + bpb_block(S12, NEW, 0.9000)
                + bpb_block(T12, NEW, 0.9040) + bpb_block(S12, OLD[S12], 0.9500))
    b = parse_bpb(log)
    assert b[(S12, NEW)] == (0.9, 1500) and b[(T12, NEW)] == (0.904, 1500) and (S12, OLD[S12]) in b
    lm = {TAGS[S12]: lm_results(0.470), TAGS[T12]: lm_results(0.475)}
    out = reading1(b, lm, TAGS)
    assert "gap relative to softmax -0.44 % (inside the 0.5 % bar)" in out[0]
    assert "difference -0.50 points (inside the 1-point bar)" in out[1]
    assert out[-1] == "reading 1: the E = F tie reproduces at 355M"
    # BPB outside (+0.51 %), nine-task inside
    b2 = {(S12, NEW): (0.9046, 1500), (T12, NEW): (0.9000, 1500)}
    out = reading1(b2, lm, TAGS)
    assert out[-1] == "reading 1: the E = F tie does not reproduce at 355M (bits-per-byte gap +0.51 %, bar 0.5 %)"
    # nine-task outside (1.01 points), BPB inside
    lm2 = {TAGS[S12]: lm_results(0.4800), TAGS[T12]: lm_results(0.4699)}
    out = reading1(b, lm2, TAGS)
    assert out[-1].endswith("(nine-task difference +1.01 points, bar 1 point)")
    # one part missing: not scored; a docs mismatch: not scored
    assert reading1(b, {}, TAGS)[-1] == "reading 1: not scored until both parts are in"
    b3 = {(S12, NEW): (0.9, 1000), (T12, NEW): (0.904, 1500)}
    assert reading1(b3, lm, TAGS)[-1] == "reading 1: not scored until both parts are in"
    # a block whose eval names another checkpoint is refused
    bad = write("bad.out", bpb_block(S12, NEW, 0.9, evaluated=T12))
    try:
        parse_bpb(bad)
        raise AssertionError("expected SystemExit")
    except SystemExit:
        pass
    # a task missing from the battery: no nine-task mean
    r = lm_results(0.47)
    del r["boolq"]
    assert sc.nine_mean(r) is None and abs(sc.nine_mean(lm_results(0.47)) - 47.0) < 1e-9


def test_versus_11_is_descriptive():
    b = {(S12, OLD[S12]): (0.9500, 1500), (S11, OLD[S11]): (0.9400, 1500), (T11, OLD[T11]): (0.9410, 1500)}
    lm = {TAGS[S12]: lm_results(0.46, 0.900, 0.36), TAGS[S11]: lm_results(0.474, 0.919, 0.314)}
    out = versus_11(b, lm, TAGS)
    joined = "\n".join(out)
    assert "FineWeb-Edu shard_0039 bits per byte: Stieltjes 0.9400 -> 0.9500 (+0.0100); softmax 0.9410 -> - (-)" in joined
    assert "LAMBADA accuracy (%): Stieltjes 31.40 -> 36.00 (+4.60)" in joined
    assert "WikiText-2 bits per byte: Stieltjes 0.9190 -> 0.9000 (-0.0190); softmax - -> - (-)" in joined
    assert "nine-task mean (points): Stieltjes 45.62 -> 44.89 (-0.73)" in joined   # LAMBADA is one of the nine
    for word in ("better", "worse", "expected", "as pre-registered"):
        assert word not in joined


GRID = sc.GRID + sc.EXT
R3_TIE = ("as pre-registered: a second 8x tie at 355M means the new tokenizer removes the "
          "Stieltjes length lead at 8x, and the length claim stays on the GPT-2-token models")


def len_block(ck, opt, scale, alphas=GRID, flat_at=None):
    """A '=== ck' block with a ppl table and a bits-per-byte table: bowls with minimum at `opt`."""
    head = "      L        off" + "".join(f"     a={a}" for a in alphas)
    rows = []
    for L in (1024, 2048, 4096, 8192):
        o = opt[L] if isinstance(opt, dict) else opt
        vals = [round(scale * (1.5 + 10 * (a - o) ** 2), 4) for a in alphas]
        rows.append(f"   {L}     9.9999 " + " ".join(f"{v:.4f}" for v in vals))
    return (f"=== {ck}\ntokenizer: stilt-tok-32k-v1\nnope=True  alpha sweep=[{', '.join(str(a) for a in alphas)}]\n"
            + head + "\n" + "\n".join(rows) + "\ndeep-tail bits per byte (same text under every tokenizer):\n"
            + head + "\n" + "\n".join(rows) + "\nDONE\n")


def test_reading3_holds_and_fails():
    p = write("len1.out", len_block(S12, 0.3, 1.0) + len_block(T12, 0.2, 1.1))
    out = reading3([p])
    assert any(x.startswith("4x: ratio softmax / Stieltjes 1.65 / 1.5 = 1.100") for x in out)
    assert out[-1] == "reading 3 holds: softmax / Stieltjes >= 1.05 at 4x and 8x"
    q = write("len2.out", len_block(S12, 0.3, 1.0) + len_block(T12, 0.2, 1.04))
    out = reading3([q])
    assert "reading 3 does not hold: below the 1.05 margin at 4x and 8x" in out
    assert out[-1] == R3_TIE
    # two logs, one block each (one job per arm)
    a, b = write("len3a.out", len_block(S12, 0.3, 1.0)), write("len3b.out", len_block(T12, 0.2, 1.1))
    assert reading3([a, b])[-1] == "reading 3 holds: softmax / Stieltjes >= 1.05 at 4x and 8x"


def test_reading3_edges_and_extension():
    # Stieltjes optimum above 0.5: the 0.525-0.7 columns are consulted, optimum 0.6 is interior
    p = write("e1.out", len_block(S12, 0.6, 1.0) + len_block(T12, 0.2, 1.1))
    out = reading3([p])
    assert "Stieltjes 8x: best 1.5 at alpha [0.6]" in out
    assert out[-1] == "reading 3 holds: softmax / Stieltjes >= 1.05 at 4x and 8x"
    # still at 0.7 after the extension: a bound, no verdict
    p = write("e2.out", len_block(S12, 0.9, 1.0) + len_block(T12, 0.2, 1.1))
    out = reading3([p])
    assert any(x.startswith("Stieltjes 8x: best") and x.endswith("EDGE-LIMITED") for x in out)
    assert any(x.startswith("8x: edge-limited: ratio >= ") for x in out)
    assert out[-1] == "reading 3: no verdict (missing or edge-limited)"
    # softmax optimum at 0.05 (no extension downward): a bound the other way
    p = write("e3.out", len_block(S12, 0.3, 1.0) + len_block(T12, 0.0, 1.1))
    out = reading3([p])
    assert any(x.startswith("8x: edge-limited: ratio <= ") for x in out)
    # an interior optimum in 0.05-0.5 is final even if a column above 0.5 is lower
    opt = {L: 0.3 for L in (1024, 2048, 4096, 8192)}
    lines = len_block(S12, opt, 1.0).split("\n")
    i = max(k for k, x in enumerate(lines) if x.strip().startswith("8192"))
    f = lines[i].split()
    f[1 + 1 + GRID.index(0.6)] = "1.0000"             # off is f[1]; 0.6 is lower than the 0.3 optimum
    lines[i] = "   " + " ".join(f)
    p = write("e4.out", "\n".join(lines) + len_block(T12, 0.2, 1.1))
    out = reading3([p])
    assert "Stieltjes 8x: best 1.5 at alpha [0.3]" in out
    # a grid with a missing point is not scored
    short = [a for a in GRID if a != 0.25]
    p = write("e5.out", len_block(S12, 0.3, 1.0, short) + len_block(T12, 0.2, 1.1))
    out = reading3([p])
    assert "Stieltjes 4x: the 0.05-0.5 grid is incomplete; not scored" in out
    assert out[-1] == "reading 3: no verdict (missing or edge-limited)"


def test_reading3_on_the_real_124m_log():
    """The 124M tokenizer cell through this scorer: E' as Stieltjes, F' as softmax.
    The registered 124M ratios are 1.046 at 4x and 1.035 at 8x."""
    p = os.path.join(os.path.dirname(HERE), "results", "len-bpb-alpha-uniform-15638836.out")
    if not os.path.exists(p):
        print("  (skipped: no len-bpb-alpha-uniform-15638836.out)")
        return
    e = "ckpt_gpt2-stj-q4-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt"
    f = "ckpt_gpt2-sdpa-nope-qk-modern-nocap-qkpilot-lr0.001-stilt-tok-32k-v1_s0.pt"
    t = open(p).read().replace(e, S12).replace(f, T12)
    out = reading3([write("real.out", t)])
    joined = "\n".join(out)
    assert "4x: ratio softmax / Stieltjes" in joined and "= 1.046 |" in joined, joined
    assert "= 1.035 |" in joined, joined
    assert out[-1] == R3_TIE


def test_main_runs_on_an_empty_results_dir():
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, "score_stilt12.py"), "--results", D],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "reading 1: not scored until both parts are in" in r.stdout
    assert "no jobs file" in r.stdout and "no length log given" in r.stdout


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
    print("ALL OK")
