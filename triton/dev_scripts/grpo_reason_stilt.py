"""Reasoning GRPO for Stilt (plan: thesis/findings/2026-07-22-plan-
judge-rl-chat.md): hard task-solving (GSM8K + ARC-Easy + synthetic
logic) with a FUSED reward — verified final-answer correctness
(dominant, judge-independent) + local Qwen judge grading the <think>
trace (0.3 partial credit). Reuses the code-GRPO machinery.

Modes:
  --probe            k samples per task, write solvable-set json
  (default)          GRPO on the probe-gated curriculum

Usage:
  python grpo_reason_stilt.py <ckpt.pt> --logic logic_tasks_v1.json \
      [--gsm8k-n 800] [--arc-n 600] [--probe | --solvable probe.json]
"""
import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt2_stieltjes import GPT, FW_DIR  # noqa: E402
from grpo_code_stilt import (  # noqa: E402
    U, A, attn_diag, batch_mask, sample_batch, seq_logprobs)
from judge_reward import LocalJudge, gold_match  # noqa: E402

os.environ.setdefault("HF_HOME", os.path.join(FW_DIR, "hf_cache"))
os.environ.setdefault("WANDB_MODE", "online")
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import torch.nn.functional as F  # noqa: E402,F401
import wandb  # noqa: E402

DEVICE = torch.device("cuda")

INSTR = ("Think step by step inside <think> and </think>, then give "
         "your final answer on a new line starting with 'Answer:'.")


def build_prompt(task):
    q = task["question"]
    if task.get("choices"):
        q += "\n" + "\n".join(f"{lab}. {txt}" for lab, txt
                              in task["choices"])
    return U + q + "\n" + INSTR + "\n" + A


def extract_answer(text):
    text = text.split("<|user|>")[0]
    tail = text.split("</think>")[-1]
    m = re.search(r"answer\s*:\s*(.+)", tail, re.I)
    return m.group(1).strip().splitlines()[0].strip() if m else None


def extract_think(text):
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    return m.group(1).strip() if m else None


def norm_num(s):
    s = s.strip().rstrip(".").replace(",", "").replace("$", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def is_correct(pred, task):
    if pred is None:
        return False
    aliases = task.get("aliases") or [task["answer"]]
    pn = norm_num(pred)
    for a in aliases:
        an = norm_num(a)
        if an is not None and pn is not None and an == pn:
            return True
        if gold_match(pred, [a]):
            return True
    return False


def load_tasks(args):
    tasks = []
    if args.logic:
        for t in json.load(open(args.logic)):
            t["channel"] = "logic"
            tasks.append(t)
    if args.gsm8k_n:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for i, ex in enumerate(ds.select(range(args.gsm8k_n))):
            ans = ex["answer"].split("####")[-1].strip()
            tasks.append({"task_id": f"gsm8k-{i:05d}", "channel": "gsm8k",
                          "question": ex["question"], "answer": ans,
                          "aliases": [ans]})
    if args.arc_n:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        for i, ex in enumerate(ds.select(range(args.arc_n))):
            labs = ex["choices"]["label"]
            txts = ex["choices"]["text"]
            key = ex["answerKey"]
            txt = txts[labs.index(key)] if key in labs else ""
            tasks.append({"task_id": f"arc-{i:05d}", "channel": "arc",
                          "question": ex["question"],
                          "choices": list(zip(labs, txts)),
                          "answer": key, "aliases": [key, txt]})
    print("tasks:", {c: sum(1 for t in tasks if t["channel"] == c)
                     for c in ("logic", "gsm8k", "arc")}, flush=True)
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--logic", default=None)
    ap.add_argument("--gsm8k-n", type=int, default=800)
    ap.add_argument("--arc-n", type=int, default=600)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-k", type=int, default=16)
    ap.add_argument("--solvable", default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--kl-beta", type=float, default=0.03)
    ap.add_argument("--judge-weight", type=float, default=0.3)
    ap.add_argument("--entropy-floor", type=float, default=0.35)
    ap.add_argument("--diag-every", type=int, default=25)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("gpt2")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    torch.manual_seed(args.seed)
    tasks = load_tasks(args)

    if args.probe:
        model.eval()
        results = {}
        for ti, task in enumerate(tasks):
            outs, _, _ = sample_batch(model, tok, [build_prompt(task)],
                                      args.probe_k, args.max_new, cfg.ctx)
            hits = [is_correct(extract_answer(o), task) for o in outs]
            fmt = [extract_answer(o) is not None for o in outs]
            results[task["task_id"]] = {
                "channel": task["channel"],
                "solved": int(sum(hits)), "fmt_ok": int(sum(fmt))}
            if ti % 50 == 0:
                ns = sum(1 for v in results.values() if v["solved"] > 0)
                print(f"probe {ti}/{len(tasks)}: solvable {ns}",
                      flush=True)
        out = args.ckpt.replace(".pt", "_reason_probe.json")
        json.dump(results, open(out, "w"))
        by_ch = {}
        for v in results.values():
            c = by_ch.setdefault(v["channel"], [0, 0])
            c[1] += 1
            c[0] += 1 if v["solved"] > 0 else 0
        print(f"PROBE DONE: "
              + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in by_ch.items())
              + f" -> {out}", flush=True)
        return

    if args.solvable:
        keep = {k for k, v in json.load(open(args.solvable)).items()
                if v["solved"] > 0}
        tasks = [t for t in tasks if t["task_id"] in keep]
        print(f"curriculum: {len(tasks)} solvable tasks", flush=True)

    ref = GPT(cfg).to(DEVICE)
    ref.load_state_dict(blob["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    judge = LocalJudge()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.0, betas=(0.9, 0.95))
    rng = np.random.default_rng(args.seed)
    run = wandb.init(project="stieltjes-flash-attn",
                     name=f"grpo-reason-{os.path.basename(args.ckpt)}"
                          f"-{os.environ.get('SLURM_JOB_ID', 'local')}",
                     config={**vars(args), "base": blob["args"]})
    diag_ids = tok(build_prompt(tasks[0]),
                   return_tensors="pt").input_ids.to(DEVICE)
    low_ent, solved_ever = 0, set()

    # auto-resume from the highest-step _reasonN ckpt (4h chain chunks)
    import glob
    start_step = 0
    cands = []
    for p in glob.glob(args.ckpt.replace(".pt", "_reason*.pt")):
        m_ = re.search(r"_reason(\d+)\.pt$", p)
        if m_:
            cands.append((int(m_.group(1)), p))
    if cands:
        start_step, p = max(cands)
        blob_r = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(blob_r["model"])
        print(f"RESUMED from {p} (step {start_step})", flush=True)
        run.config.update({"resumed_from_step": start_step},
                          allow_val_change=True)

    for step in range(start_step, args.steps):
        t0 = time.time()
        model.eval()
        batch = [tasks[i] for i in
                 rng.integers(0, len(tasks), args.prompts_per_step)]
        prompts = [build_prompt(t) for t in batch]
        outs, x, plen = sample_batch(model, tok, prompts, args.k,
                                     args.max_new, cfg.ctx)
        correct, thinks = [], []
        for i, o in enumerate(outs):
            task = batch[i // args.k]
            c = is_correct(extract_answer(o), task)
            correct.append(float(c))
            if c:
                solved_ever.add(task["task_id"])
            thinks.append(extract_think(o) or "")
        jscores, jfails = judge.score(
            [batch[i // args.k]["question"] for i in range(len(outs))],
            [t if t else "(no reasoning given)" for t in thinks])
        jnorm = [(s or 1) / 10.0 for s in jscores]
        lens = [len(o.split()) for o in outs]
        rewards = [c + args.judge_weight * j - 0.05 * (L / args.max_new)
                   for c, j, L in zip(correct, jnorm, lens)]
        R = torch.tensor(rewards, device=DEVICE).view(-1, args.k)
        adv = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True)
                                               + 1e-6)
        adv = adv.view(-1)

        model.train()
        # microbatched logprob/backward (full-batch OOMs; see code-GRPO
        # validation run 12554946)
        opt.zero_grad(set_to_none=True)
        B = x.shape[0]
        mbs = args.micro_bs
        full_mask = batch_mask(x, plen, tok)
        M_total = full_mask.sum().clamp_min(1).float()
        pg_val = kl_val = ent_num = 0.0
        for i in range(0, B, mbs):
            x_ = x[i:i + mbs]
            m_ = full_mask[i:i + mbs]
            tok_lp, _, ent_ = seq_logprobs(model, x_, plen, tok, cfg.ctx)
            with torch.no_grad():
                ref_lp, _, _ = seq_logprobs(ref, x_, plen, tok, cfg.ctx)
            adv_ = adv[i:i + mbs]
            pg_mb = -(adv_[:, None] * tok_lp * m_).sum() / M_total
            kl_mb = ((tok_lp - ref_lp) * m_).sum() / M_total
            (pg_mb + args.kl_beta * kl_mb).backward()
            pg_val += pg_mb.item()
            kl_val += kl_mb.item()
            ent_num += (ent_ * m_).sum().item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        gen_ent = ent_num / M_total.item()
        ch_acc = {}
        for i, o in enumerate(outs):
            ch = batch[i // args.k]["channel"]
            a = ch_acc.setdefault(ch, [0, 0])
            a[0] += correct[i]
            a[1] += 1
        log = {"step": step, "reward_mean": R.mean().item(),
               "reward_std": R.std().item(),
               "verified_acc": float(np.mean(correct)),
               "judge_think_mean": float(np.mean(jnorm)),
               "judge_parse_fails": jfails,
               "fmt_rate": float(np.mean(
                   [extract_answer(o) is not None for o in outs])),
               "think_rate": float(np.mean([bool(t) for t in thinks])),
               "kl": kl_val, "pg_loss": pg_val,
               "loss": pg_val + args.kl_beta * kl_val, "grad_norm": float(gnorm),
               "entropy": gen_ent,
               "gen_len_mean": float(np.mean(lens)),
               "unique_solved_cum": len(solved_ever),
               "step_seconds": time.time() - t0}
        for ch, (c, n) in ch_acc.items():
            log[f"acc_{ch}"] = c / n
        if step % args.diag_every == 0:
            model.eval()
            log.update(attn_diag(model, cfg, diag_ids))
            model.train()
        if step % 50 == 0:
            tbl = wandb.Table(columns=["step", "reward", "completion"])
            for i in np.argsort(rewards)[-2:]:
                tbl.add_data(step, rewards[i], outs[i][:1500])
            log["samples"] = tbl
        run.log(log)
        if step % 5 == 0:
            print(f"step {step:4d} R {R.mean():.3f} acc "
                  f"{np.mean(correct):.3f} judge {np.mean(jnorm):.2f} "
                  f"KL {kl_val:.4f} ent {gen_ent:.3f}", flush=True)
        low_ent = low_ent + 1 if gen_ent < args.entropy_floor else 0
        if low_ent >= 15:
            print("ENTROPY FLOOR hit — stopping", flush=True)
            break
        if step % 50 == 0 and step > 0:
            p = args.ckpt.replace(".pt", f"_reason{step}.pt")
            torch.save({"model": model.state_dict(),
                        "args": vars(cfg)}, p)
            print(f"  ckpt {p}", flush=True)

    out = args.ckpt.replace(".pt", "_reason_final.pt")
    torch.save({"model": model.state_dict(), "args": vars(cfg),
                "reason_args": vars(args)}, out)
    print(f"FINAL {out}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
