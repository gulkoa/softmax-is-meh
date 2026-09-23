"""Fill the GAPS in one pretraining label's wandb record from its SLURM chunk
logs. Unlike wandb_backfill_from_logs.py (whole-label reconstruction), this
pushes only rows that no synced chunk run carries: whole chunks whose offline
wandb dir was lost, and the tail rows a TIMEOUT kill dropped before the
offline writer flushed. One run `<label>-s0-logbackfill` (train/loss,
val/loss, val/ppl logged at step=s); source jobs and gap steps go in config.
Login-node script. Usage:
  python wandb_backfill_gaps.py <label> [--dry-run]
"""
import argparse, glob, re
import wandb

R = "/users/PAS2402/alexg/softmax/softmax-is-meh/results"
PROJECT = "gulkoa/stieltjes-flash-attn"
GLOBS = ("gpt2-medium-*.out", "gpt2-medium2-*.out", "gpt2-rope-*.out", "nope-qk-*.out")


def parse(path):
    txt = open(path, errors="ignore").read()
    m = re.search(r"^(gpt2-\S+?): [\d.]+M params, \d+ steps", txt, re.M)
    if not m:
        return None
    train = {int(a): float(b) for a, b in re.findall(r"^step\s+(\d+)/\d+ loss ([\d.]+)", txt, re.M)}
    val = {int(a): (float(b), float(c)) for a, b, c in
           re.findall(r"\[val\] step (\d+) loss ([\d.]+) ppl ([\d.]+)", txt)}
    job = int(re.search(r"-(\d+)\.out$", path).group(1))
    return m.group(1), job, train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude-jobs", default="", help="comma-separated chunk job ids to ignore "
                    "(e.g. a diverged attempt that shares the log header label)")
    a = ap.parse_args()
    excl = {int(x) for x in a.exclude_jobs.split(",") if x}
    chunks = []
    for g in GLOBS:
        for p in glob.glob(f"{R}/{g}"):
            r = parse(p)
            if r and r[0] == a.label and (r[2] or r[3]) and r[1] not in excl:
                chunks.append(r[1:])
    chunks.sort()
    api = wandb.Api(timeout=120)
    runs = {}
    for r in api.runs(PROJECT, per_page=200):
        m = re.match(rf"^{re.escape(a.label)}-s\d+-(\d+)$", r.name)
        if m:
            runs[int(m.group(1))] = r
    synced = {}                                  # job -> (train steps, val steps) held by its wandb run
    for job, _, _ in chunks:
        r = runs.get(job)
        if r is not None:
            synced[job] = ({int(h["step"]) for h in r.scan_history(keys=["step", "train_loss"])},
                           {int(h["step"]) for h in r.scan_history(keys=["step", "val_loss"])})
    gap_train, gap_val, src = {}, {}, []
    for job, train, val in chunks:              # later chunk wins on re-run steps
        if job in synced:
            have_t, have_v = synced[job]
            why = "tail rows lost"
        else:
            # a lost chunk re-validates the resume step its predecessor already logged
            # (predecessor's value is superseded) but a LATER synced chunk's re-validation
            # supersedes this chunk's last step: leave those to the synced run
            have_t = set().union(*(t for j, (t, _) in synced.items() if j > job)) if synced else set()
            have_v = set().union(*(v for j, (_, v) in synced.items() if j > job)) if synced else set()
            why = "no wandb run"
        gt = {s: l for s, l in train.items() if s not in have_t}
        gv = {s: v for s, v in val.items() if s not in have_v}
        if gt or gv:
            src.append({"job": job, "why": why, "train_rows": len(gt), "val_steps": sorted(gv)})
            gap_train.update(gt); gap_val.update(gv)
            print(f"  {job}: {why}: {len(gt)} train rows, val steps {sorted(gv)}")
    if not gap_train and not gap_val:
        print("no gaps"); return
    print(f"{a.label}: {len(gap_train)} train rows + {len(gap_val)} val points from {len(src)} chunk log(s)")
    if a.dry_run:
        return
    run = wandb.init(entity="gulkoa", project=PROJECT.split("/")[1],
                     name=f"{a.label}-s0-logbackfill",
                     id=re.sub(r"[^a-z0-9]", "", a.label)[-22:] + "gap",
                     resume="allow",
                     config={"label": a.label, "backfill": "gaps-from-logs", "source_jobs": src,
                             "gap_val_steps": sorted(gap_val)},
                     notes="gap fill from SLURM chunk logs: chunks whose offline wandb dir was "
                           "lost and tail rows dropped by TIMEOUT kills; see config.source_jobs")
    for s in sorted(set(gap_train) | set(gap_val)):
        d = {}
        if s in gap_train: d["train/loss"] = gap_train[s]
        if s in gap_val: d["val/loss"], d["val/ppl"] = gap_val[s]
        run.log(d, step=s)
    run.finish()
    print("pushed:", run.url)


if __name__ == "__main__":
    main()
