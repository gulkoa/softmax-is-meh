"""Push ONE consolidated wandb run per 355M pretraining, x-axis = optimizer step.

Why: the 355M chains were trained by the old trainer, which called run.log()
without `step=`, so wandb's default axis for those runs is the log-call index
(one call per 20 optimizer steps) — the source of the "10M tokens per step"
and "step 9,000 vs 500" confusion in the 08-25 meeting. The new trainer logs
`step=` (merged 2026-09-15); this script gives the historical trainings the
same view: `<label>-s0-consolidated` with train/loss (every 20 steps, from
the chunk logs; a later chunk overrides the steps an earlier chunk re-ran),
val/loss + val/ppl (= exp(val/loss), from the audited curve CSV), and
`tokens` = step x 524,288 — logged with step=<optimizer step>, so `_step`
IS the optimizer step.

Derived data (presentation layer): regenerate after any change to the CSV
(`--replace` deletes an existing run of the same id first). Excluded from
dump_pretrain_curves.py by name. Login-node script.
  python wandb_push_consolidated.py [--csv <path>] [--replace] [--dry-run] [--only <label>]
"""
import argparse, collections, csv, glob, math, re
import wandb

R = "/users/PAS2402/alexg/softmax/softmax-is-meh/results"
CSV = f"{R}/pretrain_curves_2026-09-22.csv"
ENTITY, PROJECT = "gulkoa", "stieltjes-flash-attn"
TOK_PER_STEP, TOTAL_STEPS = 524_288, 28_610
GLOBS = ("gpt2-medium-*.out", "gpt2-medium2-*.out", "gpt2-rope-*.out", "nope-qk-*.out")
DIVERGED_POSITIONAL_JOBS = {12428685, 12428687}      # share the log header with the shipped 3e-4 run

COMMON = dict(total_steps=TOTAL_STEPS, tokens_per_step=TOK_PER_STEP,
              total_tokens=TOTAL_STEPS * TOK_PER_STEP, micro_bs=8, grad_accum=64, ctx=1024,
              warmup=2000, seed=0, n_layer=24, n_head=16, n_embd=1024,
              data_mix="70/15/15 FineWeb-Edu / FineMath-4+ / codeparrot", val_every=250,
              train_log_every=20, scale_learnable=True, cap=15,
              source="chunk .out logs + results/pretrain_curves_2026-09-22.csv (audited 2026-09-22)")
LABELS = {
    "gpt2-stj-q4-medium-mix": dict(model="stilt.1-355m", attn="stj", q=4, positions="learned", qk_norm=False,
                                   lr=6e-4, params=354_650_112, outcome="clean",
                                   jobs="12428686,12428688,12468726,12468727"),
    "gpt2-sdpa-medium-mix": dict(model="stilt.1-355m-softmax-baseline", attn="sdpa", positions="learned",
                                 qk_norm=False, lr=3e-4, params=354_650_112,
                                 outcome="clean; restart from step 0 after the 6e-4 attempt diverged",
                                 jobs="12456680,12456681,12524577"),
    "gpt2-sdpa-medium-mix-lr0.0006-diverged": dict(model="abandoned attempt (positional softmax @6e-4)",
                                                   attn="sdpa", positions="learned", qk_norm=False, lr=6e-4,
                                                   params=354_650_112, diverged=True,
                                                   outcome="val ppl 1,174 at 12,500; 50-640 through 15,500; abandoned",
                                                   jobs="12428685,12428687"),
    "gpt2-stj-q4-nope-nope-mix-lr0.0006": dict(model="stilt.1.1-355m-nope", attn="stj", q=4, positions="nope",
                                               qk_norm=False, lr=6e-4, params=353_601_920, outcome="clean",
                                               jobs="12586715-12586740,13246245-13246247"),
    "gpt2-sdpa-nope-nope-mix-lr0.0002": dict(model="stilt.1.1-355m-nope-softmax-baseline", attn="sdpa",
                                             positions="nope", qk_norm=False, lr=2e-4, params=353_601_920,
                                             outcome="clean; restart from step 0 after the 3e-4 attempt diverged",
                                             jobs="13302855-13302857"),
    "gpt2-sdpa-nope-nope-mix-lr0.0003": dict(model="abandoned attempt (NoPE softmax @3e-4)", attn="sdpa",
                                             positions="nope", qk_norm=False, lr=3e-4, params=353_601_920,
                                             diverged=True,
                                             outcome="spikes at 11,500 (183.5) and 13,500 (570.3), 447.6 at 14,250; cancelled",
                                             jobs="12586741,13246248-13246256"),
    "gpt2-sdpa-nope-qk-nope-mix-lr0.0006": dict(model="fair twin (softmax NoPE + QK-norm; not shipped)",
                                                attn="sdpa", positions="nope", qk_norm=True, lr=6e-4,
                                                params=353_602_688, outcome="clean", jobs="14077914-14077916"),
    "gpt2-stj-q4-nope-qk-nope-mix-lr0.0006": dict(model="symmetric arm (Stieltjes NoPE + QK-norm; not shipped)",
                                                  attn="stj", q=4, positions="nope", qk_norm=True, lr=6e-4,
                                                  params=353_602_688, outcome="clean", jobs="14326904-14326907"),
    "gpt2-stj-q4-rope-mix-lr0.0006": dict(model="stilt.1.1-355m-rope", attn="stj", q=4, positions="rope",
                                          qk_norm=False, lr=6e-4, params=353_601_920, outcome="clean",
                                          jobs="13297104-13297121,13337429-13337448,13457491-13457502,13815624-13815631"),
}


def run_id(label):
    return "c" + re.sub(r"[^a-z0-9]", "", label)[-30:]


def train_rows():
    """label -> {step: loss} from every chunk log, later chunk winning on re-run steps."""
    out = collections.defaultdict(dict)
    files = []
    for g in GLOBS:
        files += glob.glob(f"{R}/{g}")
    for p in sorted(files, key=lambda p: int(re.search(r"-(\d+)\.out$", p).group(1))):
        txt = open(p, errors="ignore").read()
        m = re.search(r"^(gpt2-\S+?): [\d.]+M params, \d+ steps", txt, re.M)
        if not m or "smoke" in m.group(1):
            continue
        lab, job = m.group(1), int(re.search(r"-(\d+)\.out$", p).group(1))
        if lab == "gpt2-sdpa-medium-mix" and job in DIVERGED_POSITIONAL_JOBS:
            lab += "-lr0.0006-diverged"
        for a, b in re.findall(r"^step\s+(\d+)/\d+ loss ([\d.]+)", txt, re.M):
            out[lab][int(a)] = float(b)
    return out


def val_rows(path):
    out = collections.defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["label"]][int(r["step"])] = float(r["val_loss"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    ids = [run_id(l) for l in LABELS]
    assert len(set(ids)) == len(ids), "run ids collide"
    train, val = train_rows(), val_rows(a.csv)
    api = wandb.Api(timeout=120)
    for label, meta in LABELS.items():
        if a.only and label != a.only:
            continue
        t, v = train[label], val[label]
        assert v, f"no val rows for {label} in {a.csv}"
        last_val = max(v)
        final_ppl = math.exp(v[last_val])
        print(f"{label}: {len(t)} train rows (to step {max(t)}), {len(v)} val rows (last {last_val}, "
              f"ppl {final_ppl:.2f}); id {run_id(label)}")
        if a.dry_run:
            continue
        if a.replace:
            try:
                api.run(f"{ENTITY}/{PROJECT}/{run_id(label)}").delete()
                print("   deleted previous run")
            except Exception:
                pass
        cfg = dict(COMMON, label=label, **meta)
        run = wandb.init(entity=ENTITY, project=PROJECT, name=f"{label}-s0-consolidated", id=run_id(label),
                         group="355m-pretrain-consolidated", job_type="consolidated",
                         tags=["consolidated", "355m"] + (["diverged"] if meta.get("diverged") else []),
                         config=cfg, resume="never",
                         notes=f"{meta['model']}: one run per training, x-axis = optimizer step; "
                               f"tokens = step x {TOK_PER_STEP:,}. Derived from the audited chunk logs / CSV "
                               f"(jobs {meta['jobs']}). See thesis finding "
                               f"2026-09-29-lm-results-discrepancies-resolved.")
        for s in sorted(set(t) | set(v)):
            d = {"tokens": s * TOK_PER_STEP}
            if s in t:
                d["train/loss"] = t[s]
            if s in v:
                d["val/loss"] = v[s]
                d["val/ppl"] = math.exp(v[s])
            run.log(d, step=s)
        run.summary["final_val_ppl"] = round(final_ppl, 2)
        run.summary["final_val_step"] = last_val
        run.summary["steps_trained"] = TOTAL_STEPS if not meta.get("diverged") else max(t)
        run.summary["tokens_trained"] = run.summary["steps_trained"] * TOK_PER_STEP
        run.finish()
        print("   pushed", run.url)


if __name__ == "__main__":
    main()
