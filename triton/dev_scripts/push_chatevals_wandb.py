"""Push chat-side eval results (deep-context recall envelopes + multiturn
back-ref summaries) to wandb as one summary run. Login-node tool;
idempotent (fixed run id, resume=allow).

Sources: <ckpt>_deepctx_recall.json on scratch; multiturn SUMMARY lines
grepped from results/*.out (they exist only in job stdout)."""
import glob, json, os, re, subprocess

SCRATCH = "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt"
RESULTS = "/users/PAS2402/alexg/softmax/softmax-is-meh/results"
ENTITY, PROJECT = "gulkoa", "stieltjes-flash-attn"


def main():
    import wandb
    rows, mt = [], {}
    for p in sorted(glob.glob(f"{SCRATCH}/*_deepctx_recall.json")):
        d = json.load(open(p))
        model = os.path.basename(p).replace("ckpt_gpt2-", "").replace(
            "_deepctx_recall.json", "")
        for r in d["rows"]:
            rows.append([model, r["depth"], r["recall"]])
    logs = glob.glob(f"{RESULTS}/code-rl-stage-1*.out")
    out = subprocess.run(
        ["grep", "-hoE",
         r"SUMMARY multiturn back-ref recall: [0-9]+/[0-9]+ = [0-9]+%\s+\(ckpt_[^)]+\)",
         *logs], capture_output=True, text=True).stdout
    for line in sorted(set(out.strip().splitlines())):
        m = re.search(r"recall: (\d+)/(\d+) .*\(ckpt_gpt2-([^)]+)\.pt\)", line)
        if m:
            mt[m.group(3)] = int(m.group(1)) / int(m.group(2))
    print(f"{len(rows)} deepctx rows, {len(mt)} multiturn summaries")
    run = wandb.init(entity=ENTITY, project=PROJECT, name="chat-evals-rollup",
                     id="chat-evals-rollup", resume="allow")
    run.log({"deepctx": wandb.Table(
        columns=["model", "depth", "recall"], data=rows)})
    run.log({"multiturn": wandb.Table(
        columns=["model", "backref_recall"],
        data=[[k, v] for k, v in sorted(mt.items())])})
    run.finish()
    print("pushed")


if __name__ == "__main__":
    main()
