"""GRPO code-RL for Stilt (plan: thesis/findings/2026-07-20-plan-rl-
coding-stilt11.md). Execution rewards via code_exec_sandbox; group-
relative advantages (no critic); PPO-clip + KL-to-reference; probe mode
builds the solvable-set curriculum.

Modes:
  --probe          sample k per train problem, write solvable-set json
  (default)        GRPO training on the solvable set (or all-train)

Usage:
  python grpo_code_stilt.py <ckpt.pt> --probe
  python grpo_code_stilt.py <ckpt.pt> [--solvable set.json]
      [--synthetic tasks.json] [--entropy-floor 0.35] [--diag-every 25]

2026-07-22 upgrades: synthetic-curriculum mixing, entropy-floor early
stop, attention-entropy instrumentation (dense recompute on a fixed
diagnostic prompt at 3 layers — the RL-dynamics mechanism readout).
"""
import argparse
import json
import math
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gpt2_stieltjes import GPT, FW_DIR  # noqa: E402
from code_exec_sandbox import reward as exec_reward  # noqa: E402

os.environ.setdefault("HF_HOME", os.path.join(FW_DIR, "hf_cache"))
os.environ.setdefault("WANDB_MODE", "offline")
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import wandb  # noqa: E402

STAGING = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "results", "hf_gpt2_staging")
sys.path.insert(0, STAGING)
from modeling_stieltjes_gpt2 import stieltjes_probs  # noqa: E402

DEVICE = torch.device("cuda")
U, A = "<|user|>\n", "<|assistant|>\n"


@torch.no_grad()
def attn_diag(model, cfg, diag_ids, layers=None):
    """Attention-entropy stats via dense recompute at a few layers on a
    fixed prompt: per-row entropy of the attention distribution (last 64
    rows). Sharpening policies -> falling attention entropy; this is the
    mechanism readout for the stj-vs-softmax RL comparison."""
    L = len(model.blocks)
    layers = layers or sorted({0, L // 2, L - 1})
    caps = {}
    hooks = []
    for li in layers:
        def keep(mod, inp, out, li=li):
            caps[li] = out.float()
        hooks.append(model.blocks[li].attn.qkv.register_forward_hook(keep))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(diag_ids)
    for h in hooks:
        h.remove()
    stats = {}
    hd = cfg.n_embd // cfg.n_head
    for li, qkv in caps.items():
        B, S, _ = qkv.shape
        q, k, _ = qkv.split(cfg.n_embd, dim=2)
        q = q.view(B, S, cfg.n_head, hd).transpose(1, 2)
        k = k.view(B, S, cfg.n_head, hd).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.full((S, S), float("-inf"), device=scores.device
                          ).triu(1)
        scores = scores + mask
        rows = scores[:, :, -64:, :]                 # last 64 query rows
        if cfg.attn == "sdpa":
            p = torch.softmax(rows, dim=-1)
        else:
            p = stieltjes_probs(rows, cfg.stj_q)
        ent = -(p.clamp_min(1e-12).log() * p).sum(-1)   # (B,H,64)
        stats[f"attn_H_l{li}_mean"] = ent.mean().item()
        stats[f"attn_H_l{li}_p10"] = ent.quantile(0.10).item()
        stats[f"attn_maxp_l{li}"] = p.amax(-1).mean().item()
    return stats


def build_prompt(problem):
    tests = "\n".join(problem["test_list"][:2])   # show 2, hold the rest out
    return (U + problem["text"].strip()
            + "\nYour code should pass these tests:\n" + tests + "\n" + A)


def extract_code(text):
    text = text.split("<|user|>")[0]
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


@torch.no_grad()
def sample_batch(model, tok, prompts, k, max_new, ctx, temperature=0.8,
                 top_k=50):
    """Returns list[list[str]] completions and (ids, gen_mask) tensors."""
    enc = [tok(p, return_tensors="pt").input_ids[0] for p in prompts]
    reps = [e for e in enc for _ in range(k)]
    Lp = max(len(e) for e in reps)
    B = len(reps)
    x = torch.full((B, Lp), tok.eos_token_id, dtype=torch.long)
    for i, e in enumerate(reps):
        x[i, Lp - len(e):] = e                   # left-pad
    x = x.to(DEVICE)
    prompt_len = Lp
    finished = torch.zeros(B, dtype=torch.bool, device=DEVICE)
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x[:, -ctx:])
        lg = logits[:, -1].float() / temperature
        v, ix = torch.topk(lg, top_k, dim=-1)
        probs = torch.softmax(v, -1)
        nxt = ix.gather(-1, torch.multinomial(probs, 1))
        nxt[finished] = tok.eos_token_id
        x = torch.cat([x, nxt], 1)
        finished |= nxt.squeeze(1) == tok.eos_token_id
        if finished.all():
            break
    outs = []
    for i in range(B):
        ids = x[i, prompt_len:]
        stop = (ids == tok.eos_token_id).nonzero()
        end = stop[0, 0].item() if len(stop) else len(ids)
        outs.append(tok.decode(ids[:end]))
    return outs, x, prompt_len


def seq_logprobs(model, x, prompt_len, tok, ctx):
    """Per-sequence sum logprob + per-token logprobs on the generated part."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x[:, -ctx:] if x.shape[1] > ctx else x)
    logits = logits.float()
    tgt = x[:, 1:]
    lp = torch.log_softmax(logits[:, :-1], dim=-1)
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, L-1)
    gen_mask = torch.zeros_like(tok_lp, dtype=torch.bool)
    gen_mask[:, prompt_len - 1:] = True
    pad_mask = tgt != tok.eos_token_id
    first_eos = (~pad_mask & gen_mask).float().cumsum(1) <= 1  # keep 1st eos
    m = gen_mask & (pad_mask | ((tgt == tok.eos_token_id) & first_eos))
    ent = -(lp.exp() * lp).sum(-1)
    return tok_lp, m, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-k", type=int, default=32)
    ap.add_argument("--solvable", default=None)
    ap.add_argument("--synthetic", default=None,
                    help="json of synthetic tasks to mix into curriculum")
    ap.add_argument("--entropy-floor", type=float, default=0.35,
                    help="stop if gen entropy stays below this 3 logs")
    ap.add_argument("--diag-every", type=int, default=25)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--kl-beta", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("gpt2")
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**blob["args"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(blob["model"])
    torch.manual_seed(args.seed)

    ds = load_dataset("google-research-datasets/mbpp", "full")
    train = [ex for ex in ds["train"]]
    print(f"MBPP train problems: {len(train)}", flush=True)

    if args.probe:
        model.eval()
        results = {}
        for pi, prob in enumerate(train):
            outs, _, _ = sample_batch(model, tok, [build_prompt(prob)],
                                      args.probe_k, args.max_new, cfg.ctx)
            fr = [exec_reward(extract_code(o), prob["test_list"],
                              prob.get("test_setup_code", ""))[1]
                  for o in outs]
            results[prob["task_id"]] = {
                "best": max(fr), "mean": float(np.mean(fr)),
                "solved": sum(f == 1.0 for f in fr)}
            if pi % 20 == 0:
                nsolv = sum(1 for v in results.values() if v["solved"] > 0)
                print(f"probe {pi}/{len(train)}: solvable so far {nsolv}",
                      flush=True)
        out = args.ckpt.replace(".pt", "_mbpp_probe.json")
        json.dump(results, open(out, "w"))
        nsolv = sum(1 for v in results.values() if v["solved"] > 0)
        print(f"PROBE DONE: {nsolv}/{len(train)} solvable at k="
              f"{args.probe_k} -> {out}", flush=True)
        return

    if args.solvable:
        keep = {int(k) for k, v in json.load(open(args.solvable)).items()
                if v["solved"] > 0}
        train = [p for p in train if p["task_id"] in keep]
        print(f"curriculum: {len(train)} solvable problems", flush=True)
    if args.synthetic:
        synth = json.load(open(args.synthetic))
        train = train + synth
        print(f"curriculum: +{len(synth)} synthetic -> {len(train)} total",
              flush=True)

    ref = GPT(cfg).to(DEVICE)
    ref.load_state_dict(blob["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.0, betas=(0.9, 0.95))
    rng = np.random.default_rng(args.seed)
    run = wandb.init(project="stieltjes-flash-attn",
                     name=f"grpo-code-{os.path.basename(args.ckpt)}"
                          f"-{os.environ.get('SLURM_JOB_ID', 'local')}",
                     config={**vars(args), "base": blob["args"]})
    diag_ids = tok(build_prompt(train[0]),
                   return_tensors="pt").input_ids.to(DEVICE)
    low_ent_logs = 0

    for step in range(args.steps):
        model.eval()
        probs_batch = [train[i] for i in
                       rng.integers(0, len(train), args.prompts_per_step)]
        prompts = [build_prompt(p) for p in probs_batch]
        outs, x, plen = sample_batch(model, tok, prompts, args.k,
                                     args.max_new, cfg.ctx)
        rewards, fracs = [], []
        for i, o in enumerate(outs):
            prob = probs_batch[i // args.k]
            r, f = exec_reward(extract_code(o), prob["test_list"],
                               prob.get("test_setup_code", ""),
                               gen_len=len(o.split()), max_len=args.max_new)
            rewards.append(r)
            fracs.append(f)
        R = torch.tensor(rewards, device=DEVICE).view(-1, args.k)
        adv = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)
        adv = adv.view(-1)

        model.train()
        tok_lp, m, ent = seq_logprobs(model, x, plen, tok, cfg.ctx)
        with torch.no_grad():
            ref_lp, _, _ = seq_logprobs(ref, x, plen, tok, cfg.ctx)
        pg = -(adv[:, None] * tok_lp * m).sum() / m.sum()
        kl = ((tok_lp - ref_lp) * m).sum() / m.sum()
        loss = pg + args.kl_beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 5 == 0:
            solved = float(np.mean([f == 1.0 for f in fracs]))
            gen_ent = (ent * m).sum().item() / m.sum().item()
            print(f"step {step:4d} R {R.mean():.3f} best {R.max():.2f} "
                  f"pass@1(batch) {solved:.3f} KL {kl.item():.4f} "
                  f"ent {gen_ent:.3f}", flush=True)
            log = {"step": step, "reward_mean": R.mean().item(),
                   "reward_max": R.max().item(), "batch_pass": solved,
                   "kl": kl.item(), "entropy": gen_ent}
            if step % args.diag_every == 0:
                model.eval()
                log.update(attn_diag(model, cfg, diag_ids))
                model.train()
            run.log(log)
            low_ent_logs = low_ent_logs + 1 \
                if gen_ent < args.entropy_floor else 0
            if low_ent_logs >= 3:
                print(f"ENTROPY FLOOR hit ({gen_ent:.3f} < "
                      f"{args.entropy_floor}) 3 logs running — stopping",
                      flush=True)
                break
        if step % 50 == 0 and step > 0:
            out = args.ckpt.replace(".pt", f"_grpo{step}.pt")
            torch.save({"model": model.state_dict(), "args": vars(cfg),
                        "grpo_args": vars(args), "grpo_step": step}, out)
            print(f"  ckpt {out}", flush=True)

    out = args.ckpt.replace(".pt", "_grpo_final.pt")
    torch.save({"model": model.state_dict(), "args": vars(cfg),
                "grpo_args": vars(args)}, out)
    print(f"FINAL {out}", flush=True)
    run.finish()


if __name__ == "__main__":
    main()
