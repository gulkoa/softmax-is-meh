"""Dry run of a launch script's data path on CPU, before any GPU job: load the
trainer's own Shards class (no model, no GPU) on the MIX and TOK the launch
script passes as --data-mix and --total-tokens, and check what the trainer
would train and validate on.

Checked:
- the launch script passes "$TOK" and "${MIX}" to the trainer;
- every source shares one tokenizer and vocab (as meta.json gives them), and
  the tokenizer file exists;
- train = every shard but the last of each source, val = the last, the two
  disjoint, and each file holds the tokens meta.json lists for it;
- the step count TOK // (micro-bs * grad-accum * ctx) (--expect-steps pins it);
- sampled train and val batches: shape and next-token shift, every id below
  the vocab, and no special id other than EOT.
Prints one ok/FAIL line per check, then ALL CHECKS PASSED or the number
failed; exit status 1 on any failure.

Usage: dryrun_trainer_data.py --launch SLURM_SCRIPT --trainer TRAINER_PY [--expect-steps N]
"""
import argparse, importlib.util, json, os, re, sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", required=True, help="slurm script that sets MIX= and TOK= and calls the trainer")
    ap.add_argument("--trainer", required=True, help="the trainer .py the launch script runs")
    ap.add_argument("--expect-steps", type=int)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()

    bad = []

    def check(ok, msg):
        print(("ok   " if ok else "FAIL ") + msg, flush=True)
        if not ok:
            bad.append(msg)

    sh = open(a.launch).read()
    mix = re.search(r'^MIX="([^"]+)"', sh, re.M).group(1)
    tok = int(re.search(r"^TOK=(\d+)", sh, re.M).group(1))

    def flag(name, default):
        m = re.search(rf"--{name}\s+(\d+)", sh)
        return int(m.group(1)) if m else default

    mbs, ga, ctx = flag("micro-bs", 8), flag("grad-accum", 64), flag("ctx", 1024)
    print(f"MIX {mix}\nTOK {tok:,}; micro-bs {mbs}, grad-accum {ga}, ctx {ctx}")
    check(a.trainer in sh, "launch script runs this trainer")
    check(re.search(r'--total-tokens\s+"\$TOK"', sh) is not None, 'launch script passes --total-tokens "$TOK"')
    check(re.search(r'--data-mix\s+"\$\{?MIX\}?"', sh) is not None, 'launch script passes --data-mix "${MIX}"')

    spec = importlib.util.spec_from_file_location("trainer_under_test", a.trainer)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    tr, va = t.Shards("train", mix), t.Shards("val", mix)

    eot, special, n_train = None, set(), 0
    for part in mix.split(","):
        d = part.split("=", 1)[1].rsplit(":", 1)[0] if "=" in part else part
        m = json.load(open(os.path.join(d, "meta.json")))
        names = [f"shard_{i:04d}.bin" for i in range(m["shards"])]
        used_tr = [os.path.basename(p) for p in tr.paths_used if os.path.dirname(p) == d]
        used_va = [os.path.basename(p) for p in va.paths_used if os.path.dirname(p) == d]
        check(used_tr == names[:-1], f"{d}: train {len(used_tr)} shards ({used_tr[0]}..{used_tr[-1]})")
        check(used_va == names[-1:], f"{d}: val {used_va}")
        sizes = [os.path.getsize(os.path.join(d, n)) for n in names]
        check(sizes == [2 * n for n in m["per_shard_tokens"]], f"{d}: file sizes = 2 x meta per_shard_tokens")
        n_train += sum(m["per_shard_tokens"][:-1])
        eot = m["eot"] if eot is None else eot
        tj = json.load(open(os.path.join(m["tokenizer_path"], "tokenizer.json")))
        special |= {x["id"] for x in tj.get("added_tokens", []) if x.get("special")}
    check(not set(tr.paths_used) & set(va.paths_used), "train and val disjoint")
    check(tr.tokenizer == va.tokenizer and tr.vocab_size == va.vocab_size,
          f"tokenizer {tr.tokenizer}, vocab {tr.vocab_size} (padded {((tr.vocab_size + 63) // 64) * 64})")
    check(tr.tokenizer_path is not None and os.path.exists(os.path.join(tr.tokenizer_path, "tokenizer.json")),
          f"tokenizer file under {tr.tokenizer_path}")
    other = sorted(special - {eot})
    print(f"     EOT {eot}; other special ids {other[:3]}..{other[-3:]} ({len(other)})" if other else f"     EOT {eot}")

    steps = tok // (mbs * ga * ctx)
    check(a.expect_steps is None or steps == a.expect_steps, f"total steps {steps:,} at {mbs * ga * ctx:,} tokens/step")
    print(f"     train tokens {n_train:,}; the budget is {tok / n_train:.1%} of them")

    rng = np.random.default_rng(0)
    for name, d in (("train", tr), ("val", va)):
        hits, eots, n = 0, 0, 0
        ok_shape, top = True, 0
        for _ in range(a.batches):
            x, y = d.batch(a.bs, ctx, rng, "cpu")
            ok_shape &= tuple(x.shape) == (a.bs, ctx) and bool((x[:, 1:] == y[:, :-1]).all())
            arr = x.numpy()
            top = max(top, int(arr.max()))
            hits += int(np.isin(arr, other).sum())
            eots += int((arr == eot).sum())
            n += arr.size
        check(ok_shape, f"{name} batches: shape ({a.bs}, {ctx}) and next-token shift")
        check(top < tr.vocab_size, f"{name} batches: largest id {top}")
        check(hits == 0, f"{name} batches: {hits} special ids other than EOT")
        print(f"     {name}: {eots / n * 1024:.2f} EOT per 1,024 tokens")

    print("ALL CHECKS PASSED" if not bad else f"{len(bad)} CHECK(S) FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
