#!/usr/bin/env python3
"""Score the 1e-3 no-norm contrast at seeds 2-4
(thesis/experiments/2026-10-18-1e-3-contrast-seeds-124m.md) from the
training logs, with the readings as pre-registered (10-18 19:45). Written
10-19, before any of its runs started.

Reads the job list ("NAME JOB [JOB ...]" per run, chunk order), merges each
run's existing chunk logs (later chunk wins on re-run steps) and flags it
with stability_flags.flags (--from 2000 --final 9500). Seeds 0 and 1 are
re-scored from the logs of the 10-18 19:44 re-score and checked against
their given verdicts (A clean at both, LS failed at both).

Verdicts follow the record: one failure event fails a run, final even while
it trains. A run that ended before the step-9,500 validation with no event
needs its stop reason: --stopped NAME=training (a crash in training:
failed) or --stopped NAME=operational (cluster, queue, an operational
cancel: no verdict; its seed leaves the paired count). Until then it is
UNRESOLVED, and every reading that needs it is withheld.

The ratio (LS / A of the step-9,500 validation perplexity) is reported
with the pre-registered phrase twice, fresh seeds first: over seeds 2-4,
then over seeds 0-4. --signature reads a probe_locate --signature output
(scripts/collapse_contrast_1e3.sh) for the collapse tallies.

    python scripts/score_contrast_1e3.py [--jobs results/contrast_1e3_seeds_jobs.txt]
        [--signature results/qk_collapse_contrast_1e3_<date>.txt]
        [--stopped NAME=training|operational ...]
"""
import argparse
import math
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_lr_headroom import attribute, ended  # noqa: E402
from stability_flags import events, flags, overwritten, parse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START, FINAL, BAR = 2000, 9500, 1.15
ARMS = ("A", "LS")
FRESH, OLD = (2, 3, 4), (0, 1)
# seeds 0 and 1: the logs of the 10-18 19:44 re-score, and their given verdicts
REF = {"A@1e-3-s0": ["14010298", "14010299"], "A@1e-3-s1": ["15516255", "15516256"],
       "LS@1e-3-s0": ["15530796"], "LS@1e-3-s1": ["15530798"]}
GIVEN = {"A@1e-3-s0": "clean", "A@1e-3-s1": "clean",
         "LS@1e-3-s0": "failed", "LS@1e-3-s1": "failed"}
STEMS = {"A": "ckpt_gpt2-stj-q4-nope-lrsweep-lr{lr}_s{seed}",
         "LS": "ckpt_gpt2-sdpa-nope-lrsweepls-lr{lr}_s{seed}"}
PROBE = re.compile(r"^probe step (\d+) M ([\d.]+) maxabs ((?:[\d.]+ ?)+)\|", re.M)
RATE = re.compile(r"^step +\d+/\d+ loss \S+ gnorm \S+ \((\d+)k tok/s\)", re.M)
SIG = re.compile(r"^SIGNATURE (\S+) step (\d+): collapsed heads per layer "
                 r"\[[^\]]*\], total (\d+); shape-only (\d+)", re.M)


def name(arm, seed):
    return f"{arm}@1e-3-s{seed}"


def status(verdict, done, stop=None):
    """clean / failed / pending / unresolved / no verdict (the record's rule)."""
    if verdict == "FAILED":
        return "failed"
    if verdict == "CLEAN":
        return "clean"
    if not done:                                    # INCOMPLETE, still queued or training
        return "pending"
    return {"training": "failed", "operational": "no verdict"}.get(stop, "unresolved")


def pair(a, ls):
    """The contrast at one seed, from the two runs' statuses."""
    if "pending" in (a, ls) or "unresolved" in (a, ls):
        return "pending"
    if "no verdict" in (a, ls):
        return "no verdict"
    if a == "clean" and ls == "failed":
        return "holds"
    if a == "failed" and ls == "clean":
        return "reversed"
    return f"no contrast (both {a})"


def arm_phrase(arm, st):
    """st: seed -> status for this arm over seeds 0-4."""
    fresh = [st[s] for s in FRESH]
    if "pending" in fresh or "unresolved" in fresh:
        return f"{arm}: pending"
    j = sum(s == "failed" for s in fresh)
    n = sum(s in ("clean", "failed") for s in fresh)
    k = j + sum(st[s] == "failed" for s in OLD)
    return (f"{arm}: {j} of {n} further seeds failed at 1e-3 "
            f"({k} of {n + 2} with seeds 0 and 1)")


def contrast_phrase(pairs):
    """pairs: seed -> pair(); the count over seeds 2-4, with seeds 0 and 1."""
    fresh = {s: pairs[s] for s in FRESH}
    if "pending" in fresh.values():
        return "contrast: pending"
    known = {s: p for s, p in fresh.items() if p != "no verdict"}
    r = sum(p == "holds" for p in known.values())
    r0 = sum(pairs[s] == "holds" for s in OLD)
    out = (f"the contrast holds at {r} of {len(known)} further seeds "
           f"({r + r0} of {len(known) + len(OLD)} with seeds 0 and 1)")
    others = [f"s{s} {p}" for s, p in sorted(fresh.items()) if p != "holds"]
    others += [f"s{s} {pairs[s]}" for s in OLD if pairs[s] != "holds"]
    return out + ("; " + ", ".join(others) if others else "")


def ratio_phrase(ratios, seeds):
    """ratios: seed -> LS / A (None = undefined); the phrase over `seeds`."""
    if any(ratios[s] == "pending" for s in seeds):
        return "ratio: pending"
    have = [s for s in seeds if ratios[s] is not None]
    q = sum(ratios[s] >= BAR for s in have)
    out = f"ratio ≥ {BAR} at {q} of {len(have)} seeds"
    und = [f"s{s}" for s in seeds if ratios[s] is None]
    return out + (f" (undefined: {', '.join(und)})" if und else "")


def sign_test(h, v):
    """Exact one-sided sign test: P(X >= h), X ~ Bin(h + v, 1/2); None if h + v = 0."""
    n = h + v
    if n == 0:
        return None
    return sum(math.comb(n, i) for i in range(h, n + 1)) / 2 ** n


def parse_signature(path, lr="0.001"):
    """run name -> {"snap": {step: total}, "final": (step, total, shape) or None}."""
    stems = {STEMS[a].format(lr=lr, seed=s): name(a, s) for a in ARMS for s in FRESH}
    out = {n: {"snap": {}, "final": None} for n in stems.values()}
    for ck, step, total, shape in SIG.findall(open(path).read()):
        base = os.path.basename(ck)
        m = re.fullmatch(r"(.+?)(?:\.step(\d+))?\.pt", base)
        if not m or m.group(1) not in stems:
            continue
        run = out[stems[m.group(1)]]
        if m.group(2):
            run["snap"][int(m.group(2))] = int(total)
        else:
            run["final"] = (int(step), int(total), int(shape))
    return out


def collapse_readings(st, first_gn5, sig):
    """st, first_gn5: run name -> status / first gradient norm > 5 (seeds 2-4)."""
    out = []
    ls = [name("LS", s) for s in FRESH]
    a = [name("A", s) for s in FRESH]
    nofinal = [n for n in ls + a if sig[n]["final"] is None]
    if nofinal:
        out.append("no final signature: " + ", ".join(nofinal))
    if any(st[n] in ("pending", "unresolved") for n in ls + a):
        return out + ["collapse: pending"]
    end = lambda n: sig[n]["final"][1] > 0          # noqa: E731
    f = [n for n in ls if st[n] == "failed" and sig[n]["final"]]
    c = [n for n in ls if st[n] == "clean" and sig[n]["final"]]
    out.append(f"{sum(end(n) for n in f)} of the {len(f)} failed LS runs end with "
               f"collapsed heads, and {sum(not end(n) for n in c)} of the {len(c)} "
               "clean LS runs have none")
    have_a = [n for n in a if sig[n]["final"]]
    out.append(f"{sum(end(n) for n in have_a)} of {len(have_a)} A runs end with "
               "collapsed heads")
    exc = ([f"{n} failed without collapsed heads" for n in f if not end(n)]
           + [f"{n} clean with {sig[n]['final'][1]} collapsed heads" for n in c if end(n)]
           + [f"{n} ends with {sig[n]['final'][1]} collapsed heads" for n in have_a if end(n)])
    out += ["exception: " + e for e in exc]
    t, notes = 0, []
    for n in f:
        snaps = sorted(s for s, k in sig[n]["snap"].items() if k > 0)
        g = first_gn5[n]
        if g is None:
            notes.append(f"{n}: no gradient norm above 5")
        elif snaps and snaps[0] < g:
            t += 1
            notes.append(f"{n}: collapse at snapshot {snaps[0]:,}, first gradient "
                         f"norm above 5 at {g:,}")
        else:
            notes.append(f"{n}: first snapshot with collapse "
                         f"{f'{snaps[0]:,}' if snaps else 'none'}, first gradient "
                         f"norm above 5 at {g:,}")
    out.append(f"in {t} of the {len(f)} failed LS runs a snapshot shows collapse "
               "before the first gradient norm above 5")
    return out + ["  " + x for x in notes]


def probe_layers(paths):
    """step -> (M, per-layer max |score|), later chunk wins; per chunk median tok/s."""
    pr, rates = {}, []
    for p in paths:
        t = open(p, errors="ignore").read()
        for s, m, per in PROBE.findall(t):
            pr[int(s)] = (float(m), [float(x) for x in per.split()])
        r = [int(x) for x in RATE.findall(t)]
        rates.append(statistics.median(r) if r else None)
    return pr, rates


def describe(n, f, val, pr, rates):
    """One descriptive line per run (layers numbered 1-12)."""
    fin = (f"{val[FINAL]:.2f} @{FINAL:,}" if FINAL in val else
           "no validation" if f["final_step"] is None else
           f"no step-{FINAL:,} validation (last {f['final_ppl']:.2f} @{f['final_step']:,})")
    lay = lambda s: pr[s][1].index(max(pr[s][1])) + 1  # noqa: E731
    d = f"{n}: final {fin}; max gradient norm after {START:,} {f['max_gn']:.2f}"
    if pr:
        on = min((s for s in pr if pr[s][0] > 1000), default=None)
        pk = max(pr, key=lambda s: pr[s][0])
        m0 = f"{pr[START][0]:.1f}" if START in pr else "-"
        d += (f"; M {m0} @{START:,}, peak {pr[pk][0]:.1f} @{pk:,} (layer {lay(pk)}), "
              f"onset {f'{on:,} (layer {lay(on)})' if on is not None else 'never'}")
    else:
        d += "; no probe"
    return d + "; median tok/s per chunk " + ", ".join(
        f"{r:g}k" if r is not None else "-" for r in rates)


def event_text(gn, loss, val, f):
    jumps = sorted(s for s, k in events(gn, loss, val, START) if k == "val jump")
    parts = []
    if f["n_gn5"]:
        parts.append(f"{f['n_gn5']} gradient norms above 5 from step {f['first_gn5']:,}")
    if jumps:
        parts.append(f"{len(jumps)} validation jump{'s' if len(jumps) > 1 else ''} "
                     f"(step{'s' if len(jumps) > 1 else ''} "
                     + ", ".join(f"{s:,}" for s in jumps) + ")")
    if f["first_bad"] is not None:
        parts.append(f"nan/inf from step {f['first_bad']:,}")
    return "; ".join(parts) if parts else "no events"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=os.path.join(ROOT, "results", "contrast_1e3_seeds_jobs.txt"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--signature", help="probe_locate --signature output")
    ap.add_argument("--sig-lr", default="0.001", help="lr in the checkpoint names")
    ap.add_argument("--stopped", action="append", default=[],
                    help="NAME=training|operational for a run that ended early with no event")
    ap.add_argument("--assume-ended", action="store_true",
                    help="treat every run as ended (tests; no sacct)")
    a = ap.parse_args()
    stop = dict(s.split("=", 1) for s in a.stopped)
    bad = {k: v for k, v in stop.items() if v not in ("training", "operational")}
    if bad:
        sys.exit(f"--stopped takes training or operational: {bad}")
    git = subprocess.run(["git", "-C", ROOT, "log", "--oneline", "-1"],
                         capture_output=True, text=True).stdout[:7]
    print(f"== scored {time.strftime('%Y-%m-%d %H:%M:%S')} at softmax-is-meh {git}")

    jobs = dict(REF)
    for line in open(a.jobs):
        if line.strip():
            n, *j = line.split()
            jobs[n] = j
    want = [name(arm, s) for arm in ARMS for s in OLD + FRESH]
    missing = [n for n in want if n not in jobs]
    if missing:
        sys.exit(f"no jobs for {missing}")

    st, fl, done_of, desc, ev, att, first_gn5 = {}, {}, {}, [], [], [], {}
    for n in want:
        logs = [os.path.join(a.results, f"qknorm-pilot-{j}.out") for j in jobs[n]]
        logs = [p for p in logs if os.path.exists(p)]
        done = done_of[n] = n in REF or a.assume_ended or ended(jobs[n])
        if not logs:
            st[n] = "pending" if not done else status("INCOMPLETE", True, stop.get(n))
            fl[n] = None
            desc.append(f"{n}: no log yet")
            continue
        gn, loss, val, probe = parse(",".join(logs))
        f = flags(gn, loss, val, probe, START, FINAL)
        st[n], fl[n] = status(f["verdict"], done, stop.get(n)), (f, val)
        first_gn5[n] = f["first_gn5"]
        if n in stop and f["verdict"] != "INCOMPLETE":
            print(f"  NOTE {n}: --stopped ignored, the log gives {f['verdict']}")
        pr, rates = probe_layers(logs)
        desc.append(describe(n, f, val, pr, rates))
        ev.append(f"{n} {st[n]}: {event_text(gn, loss, val, f)}")
        if f["missing"]:
            ev.append(f"  WARNING {n}: {f['missing']} of the 20-step log lines after "
                      f"step {START:,} are missing")
        lost = overwritten(",".join(logs), START)
        if lost:
            ev.append(f"  NOTE {n}: {len(lost)} event(s) overwritten by a later chunk's "
                      "re-run (shown, not counted): "
                      + ", ".join(f"{k} @{s}" for s, k in lost))
        if st[n] == "failed" and n not in REF:
            att.append(f"attribution {n}: {attribute(f, probe)}")
        if f["first_bad"] is not None and not done:
            ev.append(f"  EARLY CANCEL {n}: nan/inf at step {f['first_bad']}; the nan "
                      f"loop should cancel {' '.join(jobs[n])}")

    print("\n-- runs (events after step 2,000)")
    for line in ev:
        print(line)
    for n, g in GIVEN.items():
        if st[n] != g:
            print(f"  MISMATCH {n}: re-scored {st[n]}, given {g}")
    print("\n-- descriptive")
    for line in desc:
        print(line)

    print("\n-- readings")
    for arm in ARMS:
        print(arm_phrase(arm, {s: st[name(arm, s)] for s in OLD + FRESH}))
    pairs = {s: pair(st[name("A", s)], st[name("LS", s)]) for s in OLD + FRESH}
    print(contrast_phrase(pairs))
    ratios = {}
    for s in OLD + FRESH:
        fa, fs = fl[name("A", s)], fl[name("LS", s)]
        va = fa[1].get(FINAL) if fa else None
        vs = fs[1].get(FINAL) if fs else None
        if va is not None and vs is not None:
            ratios[s] = vs / va
        elif not (done_of[name("A", s)] and done_of[name("LS", s)]):
            ratios[s] = "pending"
        else:
            ratios[s] = None
    print("ratio per seed (LS / A, step-9,500 validation perplexity): " + ", ".join(
        f"s{s} " + (f"{r:.3f}" if isinstance(r, float) else str(r or "undefined"))
        for s, r in ratios.items()))
    print("seeds 2-4: " + ratio_phrase(ratios, FRESH))
    print("seeds 0-4: " + ratio_phrase(ratios, OLD + FRESH))
    if "pending" in pairs.values():
        print("sign test: pending")
    else:
        h = sum(p == "holds" for p in pairs.values())
        v = sum(p == "reversed" for p in pairs.values())
        p = sign_test(h, v)
        print(f"sign test (exact, one-sided, discordant seeds of 0-4): {h} holds, "
              f"{v} reversed, p = {'undefined' if p is None else f'{p:.5g}'}; "
              "seeds 0 and 1 were known when this extension was chosen")
    for line in att:
        print(line)
    if a.signature:
        print(f"\n-- collapse signature ({os.path.basename(a.signature)})")
        sig = parse_signature(a.signature, a.sig_lr)
        for n in [name(arm, s) for arm in ARMS for s in FRESH]:
            fin = sig[n]["final"]
            snaps = ", ".join(f"{s:,}:{k}" for s, k in sorted(sig[n]["snap"].items()))
            print(f"{n}: final " + (f"{fin[1]} collapsed, {fin[2]} shape-only @{fin[0]:,}"
                                    if fin else "none") + f"; snapshots {snaps or 'none'}")
        fresh = {name(arm, s) for arm in ARMS for s in FRESH}
        for line in collapse_readings({n: st[n] for n in fresh},
                                      {n: first_gn5.get(n) for n in fresh}, sig):
            print(line)


if __name__ == "__main__":
    main()
