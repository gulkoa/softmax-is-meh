"""Log pass@k eval JSONs to wandb (the GPU evaluator writes JSON only;
this login-node script puts the aggregates next to the GRPO curves).
Idempotent-ish: run after each eval harvest; one wandb run per JSON,
named codeeval-<ckpt-stem>-<dataset>.

Usage: python log_codeeval_wandb.py <codeeval json> [...]
"""
import json
import os
import re
import sys

import wandb

for path in sys.argv[1:]:
    blob = json.load(open(path))
    base = os.path.basename(path)
    m = re.match(r"(.+)_codeeval_(\w+)\.json", base)
    if not m:
        m = re.match(r"codeeval_(.+)_(\w+)\.json", base)
    stem, dataset = m.group(1), m.group(2)
    run = wandb.init(project="stieltjes-flash-attn",
                     name=f"codeeval-{stem}-{dataset}",
                     config={"source_json": path, "dataset": dataset,
                             "ckpt": stem},
                     reinit=True)
    run.summary.update({f"{dataset}_{k}": v
                        for k, v in blob["agg"].items()})
    run.summary["n_problems"] = len(blob["per_problem"])
    run.finish()
    print(f"logged {run.name}: {blob['agg']}")
