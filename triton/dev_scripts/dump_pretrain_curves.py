"""Dump the validation loss/ppl curve of EVERY pretraining run in wandb,
merging resume chunks (and log-backfilled runs) into one series per
training label — deterministic precedence: later chunk > earlier chunk >
log-backfill run, so re-validated resume steps take the trajectory the
final model actually followed. Writes results/pretrain_curves_<date>.csv and prints a
markdown table: val loss/ppl at 10/25/50/75/100% of the planned steps
plus the final logged point.  Login-node script (wandb online API).

Usage: python dump_pretrain_curves.py [--out results/pretrain_curves.csv]
"""
import argparse
import collections
import csv
import datetime as dt
import math
import re

import wandb

PROJECT = "gulkoa/stieltjes-flash-attn"
FRACS = (0.10, 0.25, 0.50, 0.75, 1.00)


def label_of(name):
    m = re.match(r"^(.*)-s\d+-(\d+|local|logbackfill)$", name)
    return m.group(1) if m else re.sub(r"-s\d+$", "", name)


def val_rows(run):
    # trainer chunks: {step, val_loss}; wandb-steps trainer: {step, val/loss};
    # log-backfilled runs: {val/loss} logged with step=s (so _step == step)
    for keys, sk, k in ((["step", "val_loss"], "step", "val_loss"),
                        (["step", "val/loss"], "step", "val/loss"),
                        (["_step", "val/loss"], "_step", "val/loss")):
        try:
            rows = [(int(h[sk]), float(h[k]))
                    for h in run.scan_history(keys=keys)
                    if h.get(k) is not None]
        except Exception:
            rows = []
        if rows:
            return rows
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or (f"/users/PAS2402/alexg/softmax/softmax-is-meh/results/"
                    f"pretrain_curves_{dt.date.today().isoformat()}.csv")
    api = wandb.Api(timeout=120)
    series = collections.defaultdict(dict)
    meta = {}
    def precedence(r):
        # whole-label log reconstructions rank lowest (real chunk runs override them);
        # a later chunk overrides the steps an earlier chunk re-ran after resume; a
        # gap backfill (wandb_backfill_gaps.py) ranks highest — by construction it
        # only holds rows no synced run carries, or re-validations that supersede them
        if r.name.endswith("-logbackfill"):
            return (2, 0) if (r.config or {}).get("backfill") == "gaps-from-logs" else (0, 0)
        m = re.search(r"-(\d+)$", r.name)
        return (1, int(m.group(1)) if m else 0)
    runs = [r for r in api.runs(PROJECT, per_page=200)
            if ((r.config or {}).get("total_steps") or r.name.endswith("-logbackfill"))
            and not r.name.endswith("-consolidated")]   # derived presentation runs, not a source
    for r in sorted(runs, key=precedence):
        cfg = r.config or {}
        lab = label_of(r.name)
        rows = val_rows(r)
        for s, l in rows:
            series[lab][s] = l
        m = meta.setdefault(lab, {"runs": [], "params": None, "total_steps": None,
                                  "lr": None, "attn": None, "tps": None})
        m["runs"].append(r.name)
        for k, ck in (("params", "params"), ("total_steps", "total_steps"),
                      ("lr", "lr"), ("attn", "attn"), ("tps", "tokens_per_step")):
            if m[k] is None and cfg.get(ck) is not None:
                m[k] = cfg[ck]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "step", "val_loss", "val_ppl"])
        for lab in sorted(series):
            for s in sorted(series[lab]):
                w.writerow([lab, s, f"{series[lab][s]:.4f}",
                            f"{math.exp(series[lab][s]):.3f}"])
    print(f"wrote {out}\n")
    hdr = ("| model (label) | params | lr | steps logged / planned | "
           + " | ".join(f"{int(x*100)}%" for x in FRACS)
           + " | final step: loss → ppl |")
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    order = sorted(series, key=lambda l: (-(meta[l]["params"] or 0), l))
    for lab in order:
        pts = series[lab]
        m = meta[lab]
        tot = m["total_steps"] or max(pts)
        if not pts or tot < 1000:        # skip 100-step throughput benches
            continue
        cells = []
        for fr in FRACS:
            tgt = fr * tot
            cand = [s for s in pts if abs(s - tgt) <= max(300, 0.02 * tot)]
            if cand:
                s = min(cand, key=lambda s: abs(s - tgt))
                cells.append(f"{pts[s]:.3f}/{math.exp(pts[s]):.2f}")
            else:
                cells.append("—")
        last = max(pts)
        pm = f"{m['params']/1e6:.0f}M" if m["params"] else "?"
        print(f"| {lab} | {pm} | {m['lr'] if m['lr'] is not None else '?'} | "
              f"{len(pts)} / {tot} | " + " | ".join(cells)
              + f" | {last}: {pts[last]:.4f} → {math.exp(pts[last]):.2f} |")


if __name__ == "__main__":
    main()
