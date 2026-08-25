"""Reconstruct a pretraining wandb run from SLURM chunk logs (for runs
whose offline wandb dirs were lost — 2026-08-06 quota prune). Parses
`step N loss L` and `[val] step N loss L ppl P` lines from every log
whose header line matches the label, dedups by global step (chunks
overlap on resume), and pushes one run `<label>-s0-logbackfill`."""
import argparse, glob, re
import wandb

R = "/users/PAS2402/alexg/softmax/softmax-is-meh/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--glob", default="gpt2-medium2-*.out")
    a = ap.parse_args()
    train, val = {}, {}
    files = 0
    for p in sorted(glob.glob(f"{R}/{a.glob}")):
        txt = open(p, errors="ignore").read(200_000_000)
        if not txt.startswith(a.label + ":") and f"\n{a.label}:" not in txt[:2000]:
            continue
        files += 1
        for m in re.finditer(r"^step\s+(\d+)/\d+ loss ([\d.]+)", txt, re.M):
            train[int(m[1])] = float(m[2])
        for m in re.finditer(r"\[val\] step (\d+) loss ([\d.]+) ppl ([\d.]+)", txt):
            val[int(m[1])] = (float(m[2]), float(m[3]))
    print(f"{files} chunk logs, {len(train)} train points, {len(val)} val points")
    if not train and not val:
        return
    run = wandb.init(entity="gulkoa", project="stieltjes-flash-attn",
                     name=f"{a.label}-s0-logbackfill",
                     id=re.sub(r"[^a-z0-9]", "", a.label)[-24:] + "bf",
                     resume="allow",
                     notes="reconstructed from SLURM chunk logs; original "
                           "offline wandb dirs lost in 2026-08-06 quota prune")
    for s in sorted(set(train) | set(val)):
        d = {}
        if s in train: d["train/loss"] = train[s]
        if s in val: d["val/loss"], d["val/ppl"] = val[s]
        run.log(d, step=s)
    run.finish()
    print("backfilled:", run.url)


if __name__ == "__main__":
    main()
