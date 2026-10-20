#!/usr/bin/env python3
"""Score the stilt.1.2 355M pair as pre-registered in
thesis/findings/2026-10-14-plan-stilt12-355m-pair.md: readings 1-3 with the
2026-10-20 amendment (FineWeb only) and the scoring note of 2026-10-20,
written before any run (login node). Reading 4 (passkey) has no threshold
and is not scored here.

Checkpoints. The tok32k trainer's label with the launch copy's flags
(slurm/stilt12_pair_v2_h100.sh: --nope --qk-norm --modern --scale-cap 0
--scale-learnable --tag stilt12 --lr 1e-3, seed 0, stilt-tok-32k-v1 shards)
is gpt2-<stj-q4|sdpa>-nope-qk-modern-nocap-stilt12-lr0.001-stilt-tok-32k-v1,
saved as ckpt_<label>_s0.pt: S12 and T12 below. S11 and T11 are the
stilt.1.1 NoPE + QK-norm pair.

Reading 1, within the pair (pass/fail). Bits per byte on the first 1,500
documents of the new validation shard (eval_bpb.py, the `web` row as
printed); the gap relative to softmax, 100 * (S / T - 1), must be within
0.5 %. The nine-task mean (lm_eval_table's nine tasks, unrounded
`acc,none`, in points) must be within 1 point. Both inside -> "the E = F
tie reproduces at 355M"; otherwise each bar that fails, with its size.
Not scored until both parts are in. The trainer's last validation is
printed with reading 2, for reference only.

Reading 1 against stilt.1.1 (descriptive: no threshold, no direction
words). Bits per byte on the old FineWeb-Edu validation shard (shard_0039
under each tokenizer, the same documents), WikiText-2 bits per byte,
LAMBADA accuracy and the nine-task mean, with stilt.1.2 - stilt.1.1 per arm.

Reading 2 (pass/fail per arm). Chunk logs merged in submission order (on a
re-run step the later chunk wins). A run is complete when a chunk logged
"FINAL ckpt:" and a validation at step 26,500 (the last one: 26,742 steps,
a validation every 250); an incomplete run is not scored and nothing about
its curve is printed. Monotone = no validation perplexity above 1.05 x the
previous validation after step 2,000, and none non-finite (the validation
rule of stability_flags.py). Gradient norms above 5 are printed as
description only: the plan's rule has only the validation clause.

Reading 3 (pass/fail). Deep-tail bits per byte on the same PG-19 text
(eval_longctx_scaled.py --text-chars 2400000): each arm's best over alpha
0.05-0.5 every 0.025, as printed. An optimum only at 0.5 is looked up again
over the 0.525-0.7 columns of the same log (pre-authorised by the scoring
note); an optimum at 0.05, or at 0.7 after that, is edge-limited: a bound,
no verdict phrase. The ratio softmax / Stieltjes at 4x and 8x is compared
with 1.05; reading 3 holds when both reach it.

Inputs:
  --jobs    the stdout of slurm/submit_stilt12_pair_v2.sh ("<stj|sdpa> chunk
            <i>: <id>" lines; chunks appended later go in the same format).
            The smoke line is skipped: the smoke writes to the same
            results/stilt12-<id>.out pattern, with a validation every 10 steps.
  --bpb     the log of slurm/stilt12_bpb_h100.sh ("=== <ckpt> | <mix>" blocks).
  --lm-root lm_eval results (results/lm_eval/<tag>/); --lm12 sets the
            stilt.1.2 tags.
  --length  the log(s) of slurm/stilt12_len_bpb_h100.sh ("=== <ckpt>" blocks).

    python scripts/score_stilt12.py [--jobs F] [--bpb LOG] [--length LOG ...]
"""
import argparse
import importlib.util
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_len_bpb_alpha_grid import parse_two_tables, verdict as len_verdict  # noqa: E402
from score_length_alpha_grid import best  # noqa: E402
from stability_flags import flags, parse  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "lm_eval_table", os.path.join(os.path.dirname(HERE), "triton", "dev_scripts", "lm_eval_table.py"))
lm_eval_table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm_eval_table)

S12 = "ckpt_gpt2-stj-q4-nope-qk-modern-nocap-stilt12-lr0.001-stilt-tok-32k-v1_s0.pt"
T12 = "ckpt_gpt2-sdpa-nope-qk-modern-nocap-stilt12-lr0.001-stilt-tok-32k-v1_s0.pt"
S11 = "ckpt_gpt2-stj-q4-nope-qk-nope-mix-lr0.0006_s0.pt"
T11 = "ckpt_gpt2-sdpa-nope-qk-nope-mix-lr0.0006_s0.pt"
NEW = "web=/fs/scratch/PAS2836/alexg/fineweb_100bt_tok32k:1.0"
OLD = {S12: "web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt_tok32k:1.0",
       T12: "web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt_tok32k:1.0",
       S11: "web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:1.0",
       T11: "web=/fs/scratch/PAS2836/alexg/fineweb_edu_10bt:1.0"}
TAGS = {S12: "stilt.1.2-355m", T12: "stilt.1.2-355m-softmax-baseline",
        S11: "stilt.1.1-355m-nope-qk", T11: "stilt.1.1-355m-nope-qk-softmax-baseline"}
DOCS = 1500
START, LAST_VAL = 2000, 26500
NINE = [k for k, _ in lm_eval_table.TASKS if k not in ("wikitext", "mmlu", "commonsense_qa")]
LO, HI, XHI, STEP = 0.05, 0.5, 0.7, 0.025
GRID = [round(LO + STEP * i, 3) for i in range(19)]          # 0.05 .. 0.5
EXT = [round(HI + STEP * (i + 1), 3) for i in range(8)]      # 0.525 .. 0.7


# ---- reading 2 -----------------------------------------------------------

def chunk_logs(jobs_file, results):
    """{arm: [results/stilt12-<id>.out, ...]} in submission order; the smoke line is skipped."""
    arms = {"stj": [], "sdpa": []}
    for line in open(jobs_file):
        m = re.match(r"^(stj|sdpa) chunk \d+: (\d+)\s*$", line)
        if m:
            arms[m.group(1)].append(os.path.join(results, f"stilt12-{m.group(2)}.out"))
    return arms


def stability(paths):
    """(lines, monotone) for one arm; monotone is None while the run is incomplete."""
    have = [p for p in paths if os.path.exists(p)]
    if not have:
        return ["no chunk log yet; not scored"], None
    gn, loss, val, probe = parse(",".join(have))
    final = any("FINAL ckpt:" in open(p, errors="ignore").read() for p in have)
    if not (final and LAST_VAL in val):
        return [f"incomplete (last validation at step {max(val) if val else None}); not scored"], None
    vs = sorted(val)
    jumps = [(a, b) for a, b in zip(vs, vs[1:]) if b > START and not val[b] <= 1.05 * val[a]]
    f = flags(gn, loss, val, probe, START, LAST_VAL)
    lines = [f"complete: validation perplexity {val[LAST_VAL]:.2f} at step {LAST_VAL} (reference only)",
             f"descriptive: {f['n_gn5']} logged gradient norms above 5 after step {START}"
             f" (first {f['first_gn5'] or '-'}), largest {f['max_gn']:.2f}"]
    if f["missing"]:
        lines.append(f"WARNING: {f['missing']} of the 20-step log lines after step {START} are missing")
    if not jumps:
        return lines + ["monotone at 1e-3 (no validation perplexity above 1.05 x the previous after "
                        "step 2,000)"], True
    a, b = jumps[0]
    return lines + [f"not monotone at 1e-3: {len(jumps)} validation jump(s) after step 2,000, first at "
                    f"step {b} (perplexity {val[a]:.2f} -> {val[b]:.2f})"], False


def reading2(mono):
    """The plan's sentences from {'stj': bool|None, 'sdpa': bool|None}; None = not complete.
    An arm's spike sentence stands once that arm is complete."""
    out = []
    if mono["sdpa"] is False:
        out.append("reading 2: a softmax spike at 1e-3 despite QK-norm, a new result (as pre-registered)")
    if mono["stj"] is False:
        out.append("reading 2: a Stieltjes spike at 1e-3, which retracts \"intrinsic stability\" at 1e-3 "
                   "(as pre-registered)")
    if mono["stj"] and mono["sdpa"]:
        out.append("reading 2: both arms monotone at 1e-3")
    if None in mono.values():
        out.append("reading 2: incomplete until both runs are complete")
    return out


# ---- reading 1 -----------------------------------------------------------

def parse_bpb(path):
    """{(ckpt, mix): (web bits per byte as printed, docs)} from '=== <ckpt> | <mix>' blocks."""
    out, key, docs = {}, None, None
    for line in open(path):
        m = re.match(r"=== (\S+) \| (\S+)\s*$", line)
        if m:
            key, docs = (m.group(1), m.group(2)), None
            continue
        m = re.match(r"ckpt (\S+) \|.*\| (\d+) docs per source", line)
        if m and key:
            if m.group(1) != key[0]:
                raise SystemExit(f"{path}: the block for {key[0]} evaluated {m.group(1)}")
            docs = int(m.group(2))
            continue
        f = line.split()
        if key and len(f) == 6 and f[0] == "web":
            out[key] = (float(f[5]), docs)
    return out


def nine_mean(r):
    """Nine-task mean accuracy in points, unrounded; None when a task is missing."""
    if r is None or not all(k in r for k in NINE):
        return None
    return 100 * sum(r[k]["acc,none"] for k in NINE) / len(NINE)


def reading1(bpb, lm, tags):
    lines, ok = [], {}
    s, t = bpb.get((S12, NEW)), bpb.get((T12, NEW))
    if s and t:
        if s[1] != DOCS or t[1] != DOCS:
            lines.append(f"bits per byte: documents {s[1]} / {t[1]}, not {DOCS}; not scored")
        else:
            gap = 100 * (s[0] / t[0] - 1)
            ok["bpb"] = (abs(gap) <= 0.5, f"bits-per-byte gap {gap:+.2f} %, bar 0.5 %")
            lines.append(f"bits per byte, new FineWeb validation shard: Stieltjes {s[0]:.4f}, softmax "
                         f"{t[0]:.4f}, gap relative to softmax {gap:+.2f} % "
                         f"({'inside' if ok['bpb'][0] else 'outside'} the 0.5 % bar)")
    else:
        lines.append("bits per byte on the new validation shard: not in yet")
    ms, mt = nine_mean(lm.get(tags[S12])), nine_mean(lm.get(tags[T12]))
    if ms is not None and mt is not None:
        d = ms - mt
        ok["nine"] = (abs(d) <= 1.0, f"nine-task difference {d:+.2f} points, bar 1 point")
        lines.append(f"nine-task mean: Stieltjes {ms:.2f}, softmax {mt:.2f}, difference {d:+.2f} points "
                     f"({'inside' if ok['nine'][0] else 'outside'} the 1-point bar)")
    else:
        lines.append("nine-task mean: not in yet")
    if len(ok) < 2:
        return lines + ["reading 1: not scored until both parts are in"]
    if all(v[0] for v in ok.values()):
        return lines + ["reading 1: the E = F tie reproduces at 355M"]
    fails = "; ".join(v[1] for v in ok.values() if not v[0])
    return lines + [f"reading 1: the E = F tie does not reproduce at 355M ({fails})"]


def measures(ck, bpb, lm, tags):
    """The four cross-tokenizer measures of one checkpoint (None where missing)."""
    b = bpb.get((ck, OLD[ck]))
    r = lm.get(tags[ck])
    return {"FineWeb-Edu shard_0039 bits per byte": b[0] if b and b[1] == DOCS else None,
            "WikiText-2 bits per byte": r["wikitext"]["bits_per_byte,none"] if r and "wikitext" in r else None,
            "LAMBADA accuracy (%)": 100 * r["lambada_openai"]["acc,none"] if r and "lambada_openai" in r else None,
            "nine-task mean (points)": nine_mean(r)}


def versus_11(bpb, lm, tags):
    """Descriptive table: stilt.1.1 and stilt.1.2 per arm, with the difference."""
    lines = ["against stilt.1.1 (descriptive; stilt.1.2 - stilt.1.1 per arm):"]
    m = {ck: measures(ck, bpb, lm, tags) for ck in (S11, S12, T11, T12)}
    fmt = lambda v: "-" if v is None else f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"  # noqa: E731
    for name in m[S11]:
        cells = []
        for old, new in ((S11, S12), (T11, T12)):
            a, b = m[old][name], m[new][name]
            d = "-" if a is None or b is None else f"{b - a:+.4f}" if abs(a) < 10 else f"{b - a:+.2f}"
            cells.append(f"{fmt(a)} -> {fmt(b)} ({d})")
        lines.append(f"  {name}: Stieltjes {cells[0]}; softmax {cells[1]}")
    return lines


# ---- reading 3 -----------------------------------------------------------

def _row(tab, L, alphas):
    """{alpha: value} for the grid points present in a parsed bits-per-byte row, or None if any is missing."""
    out = {}
    for g in alphas:
        hit = [a for a in tab[L] if a != "off" and abs(a - g) < 1e-9]
        if not hit:
            return None
        out[g] = tab[L][hit[0]]
    return out


def reading3(logs):
    tabs = {}
    for p in logs:
        tabs.update(parse_two_tables(p))
    lines, best_of = [], {}
    for arm, ck in (("Stieltjes", S12), ("softmax", T12)):
        if ck not in tabs:
            lines.append(f"{arm}: no length block; not scored")
            continue
        for L in (4096, 8192):
            core = _row(tabs[ck]["bpb"], L, GRID) if L in tabs[ck]["bpb"] else None
            if core is None:
                lines.append(f"{arm} {L // 1024}x: the 0.05-0.5 grid is incomplete; not scored")
                continue
            v, al, edge = best(core, LO, HI)
            limited = edge
            if edge and al == [HI]:
                ext = _row(tabs[ck]["bpb"], L, EXT)
                if ext is None:
                    lines.append(f"{arm} {L // 1024}x: optimum at 0.5 and the 0.525-0.7 columns are missing")
                else:
                    v, al, limited = best({**core, **ext}, LO, XHI)
            best_of[arm, L] = (v, limited)
            lines.append(f"{arm} {L // 1024}x: best {v} at alpha {al}{'  EDGE-LIMITED' if limited else ''}")
    verdicts = {}
    for L in (4096, 8192):
        k = L // 1024
        if ("Stieltjes", L) not in best_of or ("softmax", L) not in best_of:
            continue
        (vs, es), (vt, et) = best_of["Stieltjes", L], best_of["softmax", L]
        r = vt / vs
        if es or et:
            bound = ">=" if es and not et else "<=" if et and not es else "unbounded:"
            lines.append(f"{k}x: edge-limited: ratio {bound} {r:.3f} (no verdict phrase)")
            verdicts[k] = None
            continue
        lines.append(f"{k}x: ratio softmax / Stieltjes {vt} / {vs} = {r:.3f} | {len_verdict(r, k)}")
        verdicts[k] = r >= 1.05
    if len(verdicts) < 2 or None in verdicts.values():
        return lines + ["reading 3: no verdict (missing or edge-limited)"]
    if all(verdicts.values()):
        return lines + ["reading 3 holds: softmax / Stieltjes >= 1.05 at 4x and 8x"]
    miss = " and ".join(f"{k}x" for k, v in verdicts.items() if not v)
    out = lines + [f"reading 3 does not hold: below the 1.05 margin at {miss}"]
    if not verdicts[8]:
        out.append("as pre-registered: a second 8x tie at 355M means the new tokenizer removes the "
                   "Stieltjes length lead at 8x, and the length claim stays on the GPT-2-token models")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(HERE), "results"))
    ap.add_argument("--jobs", help="submit script stdout (default <results>/stilt12_jobs.txt)")
    ap.add_argument("--bpb", help="slurm/stilt12_bpb_h100.sh log")
    ap.add_argument("--lm-root", help="lm_eval results root (default <results>/lm_eval)")
    ap.add_argument("--lm12", nargs=2, metavar=("STJ_TAG", "SDPA_TAG"),
                    default=[TAGS[S12], TAGS[T12]])
    ap.add_argument("--length", nargs="+", default=[], help="slurm/stilt12_len_bpb_h100.sh log(s)")
    a = ap.parse_args()
    tags = {**TAGS, S12: a.lm12[0], T12: a.lm12[1]}
    jobs = a.jobs or os.path.join(a.results, "stilt12_jobs.txt")
    root = a.lm_root or os.path.join(a.results, "lm_eval")
    lm = lm_eval_table.load(root) if os.path.isdir(root) else {}
    bpb = parse_bpb(a.bpb) if a.bpb else {}
    print("== reading 1, within the pair")
    for line in reading1(bpb, lm, tags):
        print(" ", line)
    for line in versus_11(bpb, lm, tags):
        print(" ", line)
    print("== reading 2")
    mono = {"stj": None, "sdpa": None}
    if os.path.exists(jobs):
        logs = chunk_logs(jobs, a.results)
        for arm in ("stj", "sdpa"):
            lines, mono[arm] = stability(logs[arm])
            for line in lines:
                print(f"  {arm}: {line}")
    else:
        print(f"  no jobs file ({jobs})")
    for line in reading2(mono):
        print(" ", line)
    print("== reading 3")
    for line in (reading3(a.length) if a.length else ["no length log given"]):
        print(" ", line)


if __name__ == "__main__":
    main()
