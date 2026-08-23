"""GPT-2-style causal LM on FineWeb-Edu shards — Stieltjes vs softmax twin.

Side-project trainer (plan: thesis/findings/2026-07-17-plan-nl-gpt2-
stieltjes.md). GPT-2 BPE (50257, padded to 50304), learned positional
embeddings, ctx 1024. Attention arms:
  sdpa : softmax flash baseline
  stj  : fused Stieltjes kernel — q=4, normalize=True, ift_grad=True,
         Halley-3 (the stability-validated fast configuration; NO
         adaptive scale per the 2026-07-16 stability findings)
Checkpoint/resume across SLURM chunks; wandb incremental (offline).
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)   # broken on this torch/H100

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
from stieltjes_flash_attn import stieltjes_attention  # noqa: E402

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402

FW_DIR = os.environ.get("FW_DIR",
                        "/fs/scratch/PAS2836/alexg/fineweb_edu_10bt")
VOCAB = 50304          # 50257 padded for tensor-core efficiency
EOT = 50256


# ---------------------------------------------------------------------------
# Data: mmap'd uint16 shards; last shard held out for validation
# ---------------------------------------------------------------------------

class Shards:
    """One or more shard dirs with mix weights: 'dir' or
    'name=dir:weight,name2=dir2:weight2'. Last shard of each source is
    held out for validation; batches sample sources by weight."""

    def __init__(self, split, spec=None):
        spec = spec or FW_DIR
        self.sources, self.weights = [], []
        for part in spec.split(","):
            if "=" in part:
                _, rest = part.split("=", 1)
                d, w = rest.rsplit(":", 1)
                weight = float(w)
            else:
                d, weight = part, 1.0
            paths = sorted(glob.glob(os.path.join(d, "shard_*.bin")))
            assert len(paths) >= 2, f"need >=2 shards in {d}"
            use = paths[:-1] if split == "train" else paths[-1:]
            self.paths_used = getattr(self, "paths_used", [])
            self.paths_used += use
            self.sources.append([np.memmap(p, dtype=np.uint16, mode="r")
                                 for p in use])
            self.weights.append(weight)
        self.weights = np.asarray(self.weights) / sum(self.weights)

    def fingerprint(self):
        """Path + size of every shard in use — pins the val set identity
        across resume chunks (a re-prepared shard dir silently changes
        what 'val' means; item 10 trainer hygiene)."""
        return [(p, os.path.getsize(p)) for p in self.paths_used]

    def batch(self, bs, ctx, rng, device):
        xs = np.empty((bs, ctx + 1), dtype=np.int64)
        for i in range(bs):
            maps = self.sources[rng.choice(len(self.sources),
                                           p=self.weights)]
            m = maps[rng.integers(len(maps))]
            off = rng.integers(0, len(m) - ctx - 1)
            xs[i] = m[off:off + ctx + 1]
        t = torch.from_numpy(xs).to(device, non_blocking=True)
        return t[:, :-1], t[:, 1:]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Attn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_head
        self.hd = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        # learnable per-head score scale, tanh-capped (C=15) — the
        # mandatory recipe whenever scales are learnable (2026-07-18)
        if getattr(cfg, "scale_learnable", False):
            self.scale_mult = nn.Parameter(torch.zeros(cfg.n_head))
        # cap C for the learnable scale: eff = 1 + C*tanh(m/C); C=0 => uncapped
        # (eff = 1 + m). Default 15 = the 2026-07-18 recipe.
        self.scale_cap = float(getattr(cfg, "scale_cap", 15.0))
        # QK-norm (plan 2026-08-16): RMS-normalize q and k over the head dim
        # with a per-head learnable gain (init 1) BEFORE RoPE and before the
        # scale multiplier. Bounds |score| <= g_q*g_k*sqrt(hd).
        self.qk_norm = getattr(cfg, "qk_norm", False)
        if self.qk_norm:
            self.q_gain = nn.Parameter(torch.ones(cfg.n_head))
            self.k_gain = nn.Parameter(torch.ones(cfg.n_head))
        self.rope = getattr(cfg, "rope", False)
        self._rope_cache = None       # (S, cos, sin) — rebuilt on S change

    def _rope_cos_sin(self, S, device):
        if self._rope_cache is not None and self._rope_cache[0] == S:
            return self._rope_cache[1], self._rope_cache[2]
        theta = getattr(self.cfg, "rope_theta", 10000.0)
        inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2, device=device,
                                            dtype=torch.float32) / self.hd))
        t = torch.arange(S, device=device, dtype=torch.float32)
        f = torch.outer(t, inv)                        # (S, hd/2)
        cos = torch.cat((f.cos(), f.cos()), dim=-1)    # (S, hd)
        sin = torch.cat((f.sin(), f.sin()), dim=-1)
        self._rope_cache = (S, cos, sin)
        return cos, sin

    @staticmethod
    def _rms_head(x, gain, eps=1e-6):
        # x: (B, H, S, D) -> unit-RMS over D (fp32), times per-head gain
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * gain.float()[None, :, None, None]).to(x.dtype)

    @staticmethod
    def _apply_rope(x, cos, sin):
        # x: (B, H, S, D), rotate-half convention; fp32 rotation for
        # precision, cast back to input dtype
        d = x.shape[-1] // 2
        xf = x.float()
        rot = torch.cat((-xf[..., d:], xf[..., :d]), dim=-1)
        return (xf * cos[None, None] + rot * sin[None, None]).to(x.dtype)

    def forward(self, x):
        B, S, E = x.shape
        q, k, v = self.qkv(x).split(E, dim=2)
        q = q.view(B, S, self.h, self.hd).transpose(1, 2)
        k = k.view(B, S, self.h, self.hd).transpose(1, 2)
        if self.qk_norm:
            q = self._rms_head(q, self.q_gain)
            k = self._rms_head(k, self.k_gain)
        if self.rope:
            cos, sin = self._rope_cos_sin(S, x.device)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
        q = q.contiguous()
        if hasattr(self, "scale_mult"):
            C = self.scale_cap
            if C > 0:
                eff = 1.0 + C * torch.tanh(self.scale_mult / C)
            else:
                eff = 1.0 + self.scale_mult
            q = q * eff[None, :, None, None].to(q.dtype)
        k = k.contiguous()
        v = v.view(B, S, self.h, self.hd).transpose(1, 2).contiguous()
        if self.cfg.attn == "sdpa":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = stieltjes_attention(
                q, k, v, causal=True, sm_scale=1.0 / math.sqrt(self.hd),
                stieltjes_q=self.cfg.stj_q, num_iter=3, solver="halley",
                normalize=True, ift_grad=True)
        return self.proj(o.transpose(1, 2).reshape(B, S, E))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attn(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(VOCAB, cfg.n_embd)
        # positional mode: learned wpe (default) | --nope (none) | --rope
        # (rotary in attention). rope and nope both skip wpe — both are
        # architecturally context-unbounded.
        self.nope = getattr(cfg, "nope", False) or getattr(cfg, "rope", False)
        if not self.nope:
            self.wpe = nn.Embedding(cfg.ctx, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, VOCAB, bias=False)
        self.head.weight = self.wte.weight
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        # GPT-2 residual-projection scaling
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or ".mlp.2.weight" in name:
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx, targets=None):
        B, S = idx.shape
        x = self.wte(idx)
        if not self.nope:
            pos = torch.arange(S, device=idx.device)
            x = x + self.wpe(pos)[None]
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.reshape(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["sdpa", "stj"], required=True)
    ap.add_argument("--stj-q", type=float, default=4.0)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--n-embd", type=int, default=768)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=32)   # 0.5M tok/step
    ap.add_argument("--total-tokens", type=float, default=10e9)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--tag", default="")
    ap.add_argument("--nope", action="store_true")
    ap.add_argument("--rope", action="store_true",
                    help="rotary positions on q/k (no wpe; pre-kernel, "
                         "zero kernel changes; context-unbounded)")
    ap.add_argument("--rope-theta", type=float, default=10000.0,
                    dest="rope_theta")
    ap.add_argument("--scale-learnable", action="store_true",
                    dest="scale_learnable")
    ap.add_argument("--scale-cap", type=float, default=15.0,
                    dest="scale_cap",
                    help="tanh cap C on the learnable scale (0 = uncapped)")
    ap.add_argument("--qk-norm", action="store_true", dest="qk_norm",
                    help="RMS-normalize q/k per head (learnable per-head "
                         "gain) before RoPE/scale — plan 2026-08-16")
    ap.add_argument("--data-mix", default=None, dest="data_mix",
                    help="'name=dir:weight,...' shard-source mix "
                         "(default: pure FW_DIR web)")
    args = ap.parse_args()
    assert not (args.nope and args.rope), "--nope and --rope are exclusive"

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    tokens_per_step = args.micro_bs * args.grad_accum * args.ctx
    total_steps = int(args.total_tokens // tokens_per_step)
    label = (f"gpt2-{args.attn}"
             + (f"-q{args.stj_q:g}" if args.attn == "stj" else "")
             + ("-nope" if args.nope else "")
             + ("-rope" if args.rope else "")
             + ("-qk" if args.qk_norm else "")
             + ("-nocap" if (args.scale_learnable and args.scale_cap == 0)
                else "")
             + (f"-{args.tag}" if args.tag else "")
             + f"-lr{args.lr:g}")
    ckpt_path = os.path.join(FW_DIR, f"ckpt_{label}_s{args.seed}.pt")

    train_d = Shards("train", args.data_mix)
    val_d = Shards("val", args.data_mix)
    model = GPT(args).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"{label}: {nparam/1e6:.1f}M params, {total_steps} steps x "
          f"{tokens_per_step} tokens", flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.1 + 0.45 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    start_step = 0
    cum_wall = 0.0                     # seconds of train wall across chunks
    val_fp = val_d.fingerprint()
    if os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        sched.load_state_dict(blob["sched"])
        start_step = blob["step"] + 1
        cum_wall = blob.get("cum_wall", 0.0)
        prev_fp = blob.get("val_fp")
        if prev_fp is not None and prev_fp != val_fp:
            print(f"WARNING: val-set fingerprint CHANGED since last chunk!\n"
                  f"  was: {prev_fp}\n  now: {val_fp}\n"
                  f"  val curves before/after this point are not comparable.",
                  flush=True)
        print(f"RESUMED from step {start_step} "
              f"(cum wall {cum_wall/3600:.2f}h)", flush=True)

    run = wandb.init(
        project="stieltjes-flash-attn",
        name=f"{label}-s{args.seed}-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config={**vars(args), "params": nparam, "total_steps": total_steps,
                "tokens_per_step": tokens_per_step, "data": FW_DIR,
                "resumed_from": start_step,
                "gpu": torch.cuda.get_device_name(0),
                "node": os.environ.get("SLURMD_NODENAME", "local"),
                "jobid": os.environ.get("SLURM_JOB_ID", "local"),
                "val_shards": [p for p, _ in val_fp]},
    )
    chunk_t0 = time.time()
    rng = np.random.default_rng(args.seed * 100003 + start_step)
    vrng_seed = 424242

    model.train()
    t0 = time.time()
    for step in range(start_step, total_steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = train_d.batch(args.micro_bs, args.ctx, rng, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 20 == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_step * (20 if step > start_step else 1) / max(dt, 1e-9)
            print(f"step {step:6d}/{total_steps} loss {loss.item():.4f} "
                  f"gnorm {float(gnorm):.2f} ({tps/1e3:.0f}k tok/s)",
                  flush=True)
            run.log({"step": step, "train_loss": loss.item(),
                     "grad_norm": float(gnorm),
                     "tok_per_s": tps, "lr": sched.get_last_lr()[0]})
        if step % args.val_every == 0 and step > start_step:
            model.eval()
            vrng = np.random.default_rng(vrng_seed)
            with torch.no_grad():
                vl = 0.0
                for _ in range(20):
                    x, y = val_d.batch(args.micro_bs, args.ctx, vrng, device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, l = model(x, y)
                    vl += l.item()
            vl /= 20
            wall_h = (cum_wall + time.time() - chunk_t0) / 3600
            cum_tok = (step + 1) * tokens_per_step
            print(f"  [val] step {step} loss {vl:.4f} "
                  f"ppl {math.exp(vl):.2f} "
                  f"(wall {wall_h:.2f}h, {cum_tok/1e9:.2f}B tok, "
                  f"{cum_tok/1e6/max(wall_h,1e-9):.0f}M tok/GPU-h)",
                  flush=True)
            run.log({"step": step, "val_loss": vl, "cum_wall_h": wall_h,
                     "cum_tokens": cum_tok,
                     "tok_per_gpu_h": cum_tok / max(wall_h, 1e-9)})
            model.train()
        if step % args.ckpt_every == 0 and step > start_step:
            torch.save({"model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "step": step, "args": vars(args),
                        "cum_wall": cum_wall + time.time() - chunk_t0,
                        "val_fp": val_fp}, ckpt_path)
            print(f"  ckpt @ {step}", flush=True)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": total_steps - 1,
                "args": vars(args),
                "cum_wall": cum_wall + time.time() - chunk_t0,
                "val_fp": val_fp}, ckpt_path)
    print(f"FINAL ckpt: {ckpt_path}")
    run.finish()
    print("DONE")


if __name__ == "__main__":
    main()
