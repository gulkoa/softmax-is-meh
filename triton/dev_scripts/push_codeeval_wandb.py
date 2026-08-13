"""Push all codeeval JSONs to wandb as one sweep table + per-model runs.

Login-node tool (user directive 2026-08-11: "push all metrics to
wandb"). Scans scratch for <ckpt>_codeeval_{mbpp,humaneval}.json,
derives model/stage/step labels from the ckpt filename, and logs:
  - one run "code-bench-zoo-<date>" with a wandb.Table of every row
    (model, arm, stage, step, bench, pass@1, pass@10) — the ranking
    artifact;
  - per-trajectory line series (step -> pass@k) so RL curves render.
Idempotent: re-running overwrites the same-named run.

Usage: .venv/bin/python push_codeeval_wandb.py [--dry-run]
"""

import argparse
import glob
import json
import os
import re

# Sweep dir, NOT scratch: only same-protocol (limit-100, n=20) JSONs
# copied there by bench_code_zoo_h100.sh feed the ranking. Pre-sweep
# JSONs live in pre_sweep_backup/ and are excluded by the flat glob.
SCRATCH = ("/users/PAS2402/alexg/softmax/softmax-is-meh/results/"
           "bench_zoo_20260811")
ENTITY, PROJECT = "gulkoa", "stieltjes-flash-attn"

PAT = re.compile(
    r"ckpt_gpt2-(?P<arm>stj-q4|sdpa|stj)-?(?P<base>[\w.-]*?)_s0"
    r"(?P<stages>(?:-it|-it2|-mt|-code|-code-r|-dr|-s1rl)*)"
    r"(?:_(?P<traj>grpo|reason)(?P<step>\d+|_final))?"
    r"_codeeval_(?P<bench>mbpp|humaneval)\.json$"
)


def rows():
    out = []
    for path in sorted(glob.glob(f"{SCRATCH}/*_codeeval_*.json")):
        name = os.path.basename(path)
        m = PAT.search(name)
        agg = json.load(open(path)).get("agg", {})
        if not agg:
            continue
        arm = "stj" if (m and m["arm"].startswith("stj")) else (
            m and m["arm"]) or "?"
        stages = (m and m["stages"] or "").strip("-")
        traj = m and m["traj"] or ""
        step = (m["step"] if m and m["step"] and m["step"] != "_final"
                else ("final" if m and m["step"] else ""))
        base = m and m["base"] or ""
        out.append({
            "file": name, "arm": arm, "base": base, "stages": stages,
            "traj": traj, "step": step,
            "bench": m["bench"] if m else "?",
            "pass@1": agg.get("1") or agg.get("pass@1"),
            "pass@10": agg.get("10") or agg.get("pass@10"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rs = rows()
    print(f"{len(rs)} codeeval rows found")
    for r in rs:
        print(f"  {r['file']}: p@1={r['pass@1']} p@10={r['pass@10']}")
    if args.dry_run or not rs:
        return
    import wandb
    run = wandb.init(entity=ENTITY, project=PROJECT,
                     name="code-bench-zoo-2026-08-11",
                     id="code-bench-zoo-20260811", resume="allow")
    cols = list(rs[0].keys())
    run.log({"zoo": wandb.Table(columns=cols,
                                data=[[r[c] for c in cols] for r in rs])})
    # trajectory series: one metric namespace per (arm, lineage, traj, bench)
    series = {}
    for r in rs:
        if not r["traj"] or r["step"] in ("", "final"):
            continue
        key = f"{r['arm']}-{r['base']}-{r['stages']}-{r['traj']}-{r['bench']}"
        series.setdefault(key, []).append((int(r["step"]), r))
    for key, pts in series.items():
        for step, r in sorted(pts):
            run.log({f"{key}/pass@1": r["pass@1"],
                     f"{key}/pass@10": r["pass@10"],
                     f"{key}/rl_step": step})
    run.finish()
    print("wandb push complete")


if __name__ == "__main__":
    main()
