"""Analyze attention patterns across all v3 runs.

Summarizes entropy and concentration per (task, q, layer) to produce a consolidated
table for the paper. Runs on login node (no GPU needed).
"""

import csv
import os
import re
from pathlib import Path

RESULTS = Path("/users/PAS2402/alexg/softmax/softmax-is-meh/results")
PATTERN = re.compile(r"^(?P<task>sorting|bfs|binary_search|max)_"
                     r"(?:(?P<attn>softmax)|stieltjes_q(?P<q>[0-9.]+))"
                     r"(?:_v3)?(?P<ctx>_ctx\d+)?$")


def parse_run_name(name):
    m = PATTERN.match(name)
    if not m:
        return None
    return {
        "task": m["task"],
        "attn": "softmax" if m["attn"] == "softmax" else "stieltjes",
        "q": m["q"] or "-",
        "ctx": m["ctx"] or "_ctx128",
    }


def read_entropy(path):
    """Read entropy.csv, return dict {layer: [head_0..head_5]}."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            layer = int(row["layer"])
            heads = [float(row[f"head_{h}"]) for h in range(6)]
            out[layer] = heads
    return out


def summarize(runs):
    """Produce summary stats per run."""
    out = []
    for run, config in runs.items():
        run_dir = RESULTS / run / "analysis"
        ent_path = run_dir / "entropy.csv"
        conc_path = run_dir / "concentration.csv"
        if not ent_path.exists():
            continue
        entropy = read_entropy(ent_path)
        conc = read_entropy(conc_path) if conc_path.exists() else {}

        all_ent = [v for heads in entropy.values() for v in heads]
        all_conc = [v for heads in conc.values() for v in heads] if conc else []

        out.append({
            "run": run,
            "task": config["task"],
            "attn": config["attn"],
            "q": config["q"],
            "ctx": config["ctx"],
            "entropy_mean": sum(all_ent) / len(all_ent) if all_ent else 0,
            "entropy_min": min(all_ent) if all_ent else 0,
            "entropy_max": max(all_ent) if all_ent else 0,
            "conc_mean": sum(all_conc) / len(all_conc) if all_conc else 0,
            "n_layers": len(entropy),
            "n_heads": len(next(iter(entropy.values()))) if entropy else 0,
        })
    return out


def main():
    # Discover all v3 runs
    runs = {}
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        # Prefer v3 versions, but also include ctx4096 variants
        if "_v3" not in d.name and "_ctx" not in d.name and d.name not in {
            "bfs_softmax", "binary_search_softmax", "sorting_softmax", "max_softmax"
        }:
            continue
        config = parse_run_name(d.name)
        if config:
            runs[d.name] = config

    summary = summarize(runs)

    # Sort by task, then q
    def sort_key(r):
        q = float(r["q"]) if r["q"] != "-" else -1
        return (r["task"], r["ctx"], r["attn"], q)
    summary.sort(key=sort_key)

    print(f"{'Task':>14} {'Attn':>10} {'q':>5} {'Entropy':>20} {'Concentr':>10}")
    print(f"{'':>14} {'':>10} {'':>5} {'mean (min–max)':>20}")
    print("-" * 70)
    for r in summary:
        ent_str = f"{r['entropy_mean']:.2f} ({r['entropy_min']:.2f}-{r['entropy_max']:.2f})"
        conc_str = f"{r['conc_mean']:.3f}"
        print(f"{r['task']:>14} {r['attn']:>10} {r['q']:>5} {ent_str:>20} {conc_str:>10}")

    # Write CSV summary
    out_csv = RESULTS / "attention_pattern_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nWritten to {out_csv}")


if __name__ == "__main__":
    main()
