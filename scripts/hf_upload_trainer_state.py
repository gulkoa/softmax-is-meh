"""Back up trainer checkpoints that exist only on scratch to one private
Hugging Face dataset repo (user direction of 2026-11-10: no backups to other
directories on this machine; upload to the Hub instead).

Selection is explicit: the in-flight checkpoints of the three unfinished
A@1e-3 seed runs and the 1,000-step snapshots of the six 1e-3 contrast runs
(seeds 2 to 4), the pre-registered inputs of the collapse-signature reading.
The converted weights of every finished run already live in the
stilt-research-124m-<run label> model repos; this repo holds the trainer
state (weights + optimizer + args + step) that those do not.

    python scripts/hf_upload_trainer_state.py --dry      # list, sizes, quota
    python scripts/hf_upload_trainer_state.py --upload   # README + files, in commits of 5
    python scripts/hf_upload_trainer_state.py --verify   # sizes (and sha256 with --sha) against the Hub

Files already on the Hub with the same size are skipped on --upload, so the
command can be re-run after an interruption. Login node: no model is built.
"""
import argparse
import glob
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_research_cards as cards  # noqa: E402

REPO = "GulkoA/stilt-research-124m-trainer-state"
SCAN = cards.SCAN
PATTERNS = (
    "ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s[234].pt",
    "ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s[234].step*.pt",
    "ckpt_gpt2-sdpa-nope-lrsweepls-lr0.001_s[234].step*.pt",
)
BATCH = 5
PRIVATE_QUOTA = 100e9  # free account, huggingface.co/docs/hub/storage-limits


def run_label(basename):
    """ckpt_gpt2-stj-q4-nope-lrsweep-lr0.001_s2.step1000.pt -> stj-q4-nope-lrsweep-lr0.001-s2"""
    return cards.label(re.sub(r"\.step\d+(?=\.pt$)", "", basename))


def step_of(basename):
    m = re.search(r"\.step(\d+)\.pt$", basename)
    return int(m.group(1)) if m else None


def select(scan=SCAN):
    """[(local path, path in repo)], stj runs first, then by path in repo."""
    out = {}
    for pat in PATTERNS:
        for p in glob.glob(os.path.join(scan, pat)):
            b = os.path.basename(p)
            out[f"{run_label(b)}/{b}"] = p
    # the in-flight Stieltjes runs first, then the finished LS runs
    return sorted(((p, r) for r, p in out.items()), key=lambda t: (not t[1].startswith("stj-"), t[1]))


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def card(files, steps):
    rows = "\n".join(f"| `{r}` | {run_label(os.path.basename(r))} | {steps[r]:,} | {os.path.getsize(p):,} |"
                     for p, r in files)
    return f"""---
license: mit
pretty_name: Stilt research 124M trainer state (backup)
---

# stilt-research-124m-trainer-state

Trainer checkpoints of 124M research runs of the Stilt program (Stieltjes
attention as a drop-in replacement for softmax), backed up from OSC scratch
on 2026-11-10. Each file is a `torch.save` dict with the keys `model`,
`optimizer`, `args` and `step`, as written by the training script of
[softmax-is-meh](https://github.com/gulkoa/softmax-is-meh) (load with
`torch.load(path, map_location="cpu", weights_only=False)`). The converted,
loadable weights of every finished run are in the `stilt-research-124m-<run
label>` model repos; this repo holds what those do not:

- the in-flight checkpoints of the three unfinished Stieltjes runs of the
  1e-3 contrast at seeds 2 to 4 (`stj-q4-nope-lrsweep-lr0.001-s2/s3/s4`,
  the chains are held on OSC), and
- the 1,000-step snapshots of all six 1e-3 contrast runs (arms A and LS,
  seeds 2 to 4), the pre-registered inputs of the collapse-signature
  reading in `experiments/2026-10-18-1e-3-contrast-seeds-124m.md` of
  [the thesis repository](https://github.com/gulkoa/softmax-research-thesis).

Numbers and verdicts live in the thesis repository, not here.

| path | run | step | bytes |
| --- | --- | --- | --- |
{rows}
"""


def hub_files(api):
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        tree = api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True)
        return {f.path: f for f in tree if hasattr(f, "size")}
    except RepositoryNotFoundError:
        return {}


def private_usage(api):
    tot = 0
    for m in api.list_models(author="GulkoA"):
        i = api.model_info(m.id, expand=["usedStorage", "private"])
        tot += (i.used_storage or 0) if i.private else 0
    for d in api.list_datasets(author="GulkoA"):
        i = api.dataset_info(d.id, expand=["usedStorage", "private"])
        tot += (i.used_storage or 0) if i.private else 0
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=SCAN)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry", action="store_true")
    g.add_argument("--upload", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--sha", action="store_true", help="with --verify: compare sha256 too")
    a = ap.parse_args(argv)
    files = select(a.scan)
    total = sum(os.path.getsize(p) for p, _ in files)
    print(f"{len(files)} files, {total / 1e9:.1f} GB -> {REPO} (private dataset)")
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    if a.dry:
        used = private_usage(api)
        print(f"private storage used {used / 1e9:.1f} GB, after upload {(used + total) / 1e9:.1f} GB "
              f"of {PRIVATE_QUOTA / 1e9:.0f} GB")
        for p, r in files:
            print(f"  {os.path.getsize(p):>14,}  {r}")
        return 0
    if a.verify:
        have = hub_files(api)
        bad = 0
        for p, r in files:
            f = have.get(r)
            if f is None:
                print(f"  MISSING   {r}"); bad += 1; continue
            if f.size != os.path.getsize(p):
                print(f"  SIZE      {r}: hub {f.size:,} local {os.path.getsize(p):,}"); bad += 1; continue
            if a.sha:
                hub_sha = getattr(getattr(f, "lfs", None), "sha256", None) or (f.lfs or {}).get("sha256") if f.lfs else None
                if hub_sha is None:
                    print(f"  size ok, no hub sha  {r}")
                elif hub_sha != sha256(p):
                    print(f"  SHA       {r}"); bad += 1
                else:
                    print(f"  ok (sha)  {r}")
            else:
                print(f"  ok        {r}")
        print(f"{len(files) - bad} of {len(files)} verified; {bad} problems; README present: {'README.md' in have}")
        return 1 if bad else 0
    # --upload
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    have = hub_files(api)
    steps = {}
    for p, r in files:
        s = step_of(os.path.basename(r))
        steps[r] = s if s is not None else cards.load_args(p)[1]
    api.upload_file(path_or_fileobj=card(files, steps).encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="dataset", commit_message="trainer-state backup card")
    todo = [(p, r) for p, r in files if r not in have or have[r].size != os.path.getsize(p)]
    print(f"{len(files) - len(todo)} already on the Hub, {len(todo)} to upload")
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=p) for p, r in batch]
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                          commit_message=f"trainer state {i // BATCH + 1}/{(len(todo) + BATCH - 1) // BATCH}")
        for p, r in batch:
            print(f"  uploaded {os.path.getsize(p):>14,}  {r}", flush=True)
    print("upload done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
