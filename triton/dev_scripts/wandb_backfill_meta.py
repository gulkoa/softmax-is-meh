"""Backfill accurate wall-time + hardware metadata onto synced wandb runs.

Run names embed the SLURM job id (…-<jobid>); sacct provides Elapsed,
NodeList, AllocTRES. Writes per-run: summary.wall_seconds,
summary.tokens_processed (chunk), summary.tokens_per_gpu_hour, and
config.hardware{gpu,node,cpus,mem_gb,cluster,jobid,elapsed}. Multi-chunk
trainings get config.chain=<label> for grouping; chain totals printed.

Login-node script (wandb online API + sacct).
"""
import re
import subprocess
from collections import defaultdict

import wandb

PROJECT = "gulkoa/stieltjes-flash-attn"
GPU_NAME = "NVIDIA H100 80GB HBM3 (1x)"
CLUSTER = "OSC Cardinal"


def sacct(jobid):
    out = subprocess.run(
        ["sacct", "-j", str(jobid), "-n", "-P",
         "--format=Elapsed,NodeList,AllocTRES,State"],
        capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return None
    f = out[0].split("|")
    h, m, s = 0, 0, 0
    el = f[0]
    if "-" in el:                       # D-HH:MM:SS
        d, rest = el.split("-")
        h = int(d) * 24
        el = rest
    parts = [int(x) for x in el.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    secs = (h + parts[0]) * 3600 + parts[1] * 60 + parts[2]
    tres = f[2] if len(f) > 2 else ""
    cpus = re.search(r"cpu=(\d+)", tres)
    mem = re.search(r"mem=(\d+)G", tres)
    return {"elapsed_s": secs, "node": f[1] if len(f) > 1 else "?",
            "cpus": int(cpus.group(1)) if cpus else None,
            "mem_gb": int(mem.group(1)) if mem else None,
            "state": f[3] if len(f) > 3 else "?"}


def main():
    api = wandb.Api()
    chains = defaultdict(list)
    for run in api.runs(PROJECT):
        m = re.search(r"-(\d{7,9})$", run.name)
        if not m:
            continue
        jobid = m.group(1)
        info = sacct(jobid)
        if info is None or info["elapsed_s"] == 0:
            continue
        cfg_tps = run.config.get("tokens_per_step")
        start = run.config.get("resumed_from", 0) or 0
        last = run.summary.get("step")
        toks = (cfg_tps * (last - start)
                if (cfg_tps and last is not None) else None)
        run.summary["wall_seconds"] = info["elapsed_s"]
        if toks:
            run.summary["tokens_processed_chunk"] = int(toks)
            run.summary["tokens_per_gpu_hour"] = int(
                toks / (info["elapsed_s"] / 3600))
        run.config["hardware"] = {
            "gpu": GPU_NAME, "cluster": CLUSTER, "node": info["node"],
            "cpus": info["cpus"], "mem_gb": info["mem_gb"],
            "slurm_jobid": jobid, "slurm_state": info["state"],
            "elapsed_hms": f"{info['elapsed_s']//3600}h"
                           f"{(info['elapsed_s']%3600)//60:02d}m"}
        chain = re.sub(r"-s\d+-\d+$", "", run.name)
        run.config["chain"] = chain
        chains[chain].append((run.name, info["elapsed_s"], toks or 0))
        run.update()
        print(f"updated {run.name}: {info['elapsed_s']}s on {info['node']}"
              + (f", {toks/1e9:.2f}B tok" if toks else ""), flush=True)

    print("\n=== chain totals (wall / tokens) ===")
    for chain, items in sorted(chains.items()):
        tw = sum(i[1] for i in items)
        tt = sum(i[2] for i in items)
        print(f"{chain}: {tw/3600:.1f} GPU-h"
              + (f", {tt/1e9:.2f}B tokens, "
                 f"{tt/(tw/3600)/1e3:.0f}k tok/GPU-h" if tt else "")
              + f"  [{len(items)} chunk(s)]")


if __name__ == "__main__":
    main()
