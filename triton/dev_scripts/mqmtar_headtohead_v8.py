"""
MQMTAR head-to-head on the ASEntmax protocol (arXiv:2506.16640).
V8 = V7 + --beta-cap (bounded beta = cap*sigmoid instead of softplus): the
asstjd arm at lr 1e-4 trained to the transition doorstep (0.556 @20k, val
blip @15k) then collapsed mid-run (3.8 by 50k); 3e-4/1e-3 collapse
immediately. Unbounded beta x Stieltjes q-sharpening is a runaway feedback
loop that entmax-1.5's saturation tolerates — bounding beta mirrors their
tanh bound on gamma.
V7 = V6 + checkpoint saving (results/mqmtar_ckpt_*.pt) + --diag N (decode N
eval samples, print generation vs target token-by-token — the 0.55-nat
loss floor at ~41%/value-token needs inspection, not more theory).
V6 = V5 + dense NAPE-capable arms for the PE pivot: full-budget pure-NoPE
softmax hit a hard 0.55-nat loss floor with EM 0.000 everywhere (job
12351108) — MQMTAR answers are +2/+3 offset copies after a key match,
which pure NoPE cannot express robustly at this scale; their default and
published rows are NAPE. New arms (dense, NAPE-capable, eval<=8k):
  softmaxd : dense softmax, no adaptive scale (their softmax row)
  asstjd   : dense AS-Stieltjes — asentmax's exact scaffold (scale, NAPE,
             masking) with the normalized Stieltjes mapping (detached-
             lambda bisection = Jack's a19bb33 `stieltjes` semantics)
V5 = V4 + flash-arm eval capped at n=100 for L>=32k (no-KV-cache generation
made 65k evals take ~5h at n=250) + per-split eval timing.
V4 = V3 + data dir overridable via MQMTAR_DATA env (full 50M-pair set —
their exact train_size, 1 epoch at 390,625 steps — for full-budget runs).
V3 = V2 + cuDNN-SDPA backend disabled (job 12345902 crash on odd seq lens).
V2 — faithful port after line-by-line review of their reference
(synthetic/src/models/architectures/sparse_gemma.py + README repro commands):

  * beta/gamma are nn.Linear projections FROM HIDDEN STATES (d_model -> heads),
    bias=True (their MQMTAR repro sets attn_scale_proj_bias=True), default
    PyTorch init — v1 wrongly projected from per-head q vectors, zero-init.
  * log-position for query at 0-based absolute position i is log(i+2)
    ("Offset by 2 so positions start at log(2)") — v1 used log(max(i+1,2)).
  * scale s = delta + softplus(beta)*(log_pos ** (2*tanh(gamma))), delta=1,
    multiplied into the query rows (their _apply_length_scaling), on top of
    qk_scale = 1/sqrt(head_dim).
  * --nape replicates their published PE: first H/2 heads get ALiBi
    (linear slopes 1/1..1/(H/2), bias = slope*(kv-q) on the causal triangle),
    last H/2 heads are NoPE and receive the adaptive scale. Their published
    MQMTAR ASEntmax row (76.7% @1024x, full budget) is asentmax + NAPE.
    (Unsupported for asstj: the fused kernel has no additive-bias input.)

Data: their scripts/generate_data.py verbatim (abc 256, id-vocab 10k,
k_len=v_len=2, num_q=4, num_kv=0.8, train 48+-16; eval 64->65,536).
Seq2seq [BOS] src [SEP] -> 12-token answer + [EOS]; loss on target only;
metric = exact match of greedy generation.

Model (their sparse_gemma shape): decoder-only 4L/512/16H/head_dim32/FFN1024.
Optim (theirs at 1/16): AdamW lr 1e-5 b(0.9,0.99) wd 0, cosine, 24,414 steps,
warmup 1,250, batch 128, bf16 autocast.

Arms: sdpa | asentmax [--nape] | asstj.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# job 12345902: cuDNN SDPA has no execution plan for this workload's odd
# sequence lengths (~89) on H100 — fall back to flash/efficient backends
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
from stieltjes_flash_attn import stieltjes_attention  # noqa: E402

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402

DATA = os.environ.get(
    "MQMTAR_DATA",
    "/users/PAS2402/alexg/softmax/softmax-is-meh/results/mqmtar_data/"
    "3M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4")
PAD, BOS, SEP, EOS = 256, 257, 258, 259
VOCAB = 260
TRG_LEN = 12          # 4 queries x (marker + 2-token value)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_split(prefix):
    src = [np.fromstring(l, dtype=np.int16, sep=" ")
           for l in open(prefix + ".src")]
    trg = [np.fromstring(l, dtype=np.int16, sep=" ")
           for l in open(prefix + ".trg")]
    return src, trg


def make_batch(src_list, trg_list, idxs, device):
    """Pack [BOS] src [SEP] trg [EOS], left-align, pad; loss mask on trg+EOS."""
    seqs, masks = [], []
    for i in idxs:
        s, t = src_list[i], trg_list[i]
        seq = np.concatenate(([BOS], s, [SEP], t, [EOS]))
        m = np.zeros(len(seq), dtype=np.bool_)
        m[len(s) + 2:] = True          # trg tokens + EOS
        seqs.append(seq)
        masks.append(m)
    L = max(len(s) for s in seqs)
    x = np.full((len(seqs), L), PAD, dtype=np.int64)
    mm = np.zeros((len(seqs), L), dtype=np.bool_)
    for j, (s, m) in enumerate(zip(seqs, masks)):
        x[j, :len(s)] = s
        mm[j, :len(s)] = m
    x = torch.from_numpy(x).to(device)
    mm = torch.from_numpy(mm).to(device)
    return x[:, :-1], x[:, 1:], mm[:, 1:]   # inputs, targets, loss-mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_ALIBI_CACHE = {}


def alibi_bias(slopes, S, device):
    """(1,H,S,S) fp32: slope * (kv - q) on the causal triangle (their
    make_bias_tensor). Built at a rounded-up size so the 13 growing
    generation steps slice one cached tensor instead of rebuilding."""
    S_cap = (S + 255) // 256 * 256
    cached = _ALIBI_CACHE.get(device.index)
    if cached is None or cached.shape[-1] < S_cap:
        pos = torch.arange(S_cap, device=device)
        diff = torch.tril(pos[None, :] - pos[:, None]).float()   # (S,S) <= 0
        _ALIBI_CACHE.clear()      # keep at most one (they grow ~S^2)
        _ALIBI_CACHE[device.index] = slopes.view(1, -1, 1, 1) * diff
    return _ALIBI_CACHE[device.index][:, :, :S, :S]


def stieltjes_dense_probs(scores, q_order, iters=50, eps=1e-20):
    """Normalized Stieltjes attention on dense fp32 scores (last dim = keys).

    weights w_j = (lam - s_j)^(-q) with lam solved per row by bisection so
    sum(w) = 1, then p = w / sum(w). Lambda is solved under no_grad and used
    detached (Jack's a19bb33 `stieltjes`: torch.where-blocked bisection +
    renormalization — the gradient form our kernel's normalize=True mode
    reproduces). Masked entries (-1e9) get (lam-s) ~ 1e9 -> w -> 0.
    """
    K = scores.shape[-1]
    with torch.no_grad():
        smax = scores.amax(dim=-1, keepdim=True)
        lo = smax + 1e-6
        hi = smax + float(K) ** (1.0 / q_order)   # sum(hi) <= K*K^-1 = 1
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = (mid - scores).clamp_min(eps).pow(-q_order).sum(-1, keepdim=True)
            gt = f > 1.0            # sum too big -> lambda must grow
            lo = torch.where(gt, mid, lo)
            hi = torch.where(gt, hi, mid)
        lam = 0.5 * (lo + hi)
    w = (lam - scores).clamp_min(eps).pow(-q_order)
    return w / w.sum(-1, keepdim=True).clamp_min(eps)


class ArmAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_head
        self.hd = cfg.head_dim
        inner = self.h * self.hd
        self.qkv = nn.Linear(cfg.d_model, 3 * inner, bias=False)
        self.proj = nn.Linear(inner, cfg.d_model, bias=False)
        # NAPE: first half ALiBi heads, last half NoPE+scaled (their
        # get_nape_slopes 'linear' + num_scaled_heads logic)
        self.n_scaled = self.h // 2 if cfg.nape else self.h
        if cfg.nape:
            slopes = torch.cat([1.0 / torch.arange(1, self.h // 2 + 1),
                                torch.zeros(self.h - self.h // 2)])
            self.register_buffer("slopes", slopes, persistent=False)
        if cfg.arm in ("asentmax", "asstj", "asstjd"):
            # their adapt-softplus-tanh: Linear from hidden states, bias per
            # README repro (attn_scale_proj_bias=True), default init
            self.beta_proj = nn.Linear(cfg.d_model, self.n_scaled, bias=True)
            self.gamma_proj = nn.Linear(cfg.d_model, self.n_scaled, bias=True)
            self.gamma_range = 2.0
            self.delta = 1.0
            # v8: optional bound on beta. asstjd at lr 1e-4 reached the
            # transition doorstep (loss 0.556 @20k) then collapsed (3.8 by
            # 50k): unbounded softplus(beta) x Stieltjes q-sharpening is a
            # runaway loop where entmax-1.5 saturates. Bounding beta is the
            # exact analogue of their tanh bound on gamma.
            self.beta_cap = float(getattr(cfg, "beta_cap", 0.0))

    def _scale(self, x, S):
        """(B, n_scaled, S) adaptive scale from hidden states x (B,S,E)."""
        raw = self.beta_proj(x.float())
        if self.beta_cap > 0:
            beta = (self.beta_cap * torch.sigmoid(raw)).transpose(1, 2)
        else:
            beta = F.softplus(raw).transpose(1, 2)
        gamma = (self.gamma_range
                 * torch.tanh(self.gamma_proj(x.float()))).transpose(1, 2)
        # their log_position: absolute 0-based position i -> log(i+2)
        logn = torch.log(torch.arange(2, S + 2, device=x.device,
                                      dtype=torch.float32))
        return self.delta + beta * logn.view(1, 1, -1).pow(gamma)

    def forward(self, x):
        B, S, _ = x.shape
        q, k, v = self.qkv(x).split(self.h * self.hd, dim=2)
        q = q.view(B, S, self.h, self.hd).transpose(1, 2).contiguous()
        k = k.view(B, S, self.h, self.hd).transpose(1, 2).contiguous()
        v = v.view(B, S, self.h, self.hd).transpose(1, 2).contiguous()
        sm = 1.0 / math.sqrt(self.hd)

        if self.cfg.arm == "sdpa":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.cfg.arm == "asstj":
            scale = self._scale(x, S)                       # (B,H,S)
            q = (q * scale.unsqueeze(-1).to(q.dtype)).contiguous()
            o = stieltjes_attention(q, k, v, causal=True, sm_scale=sm,
                                    stieltjes_q=self.cfg.stj_q, num_iter=8,
                                    normalize=True)
        elif self.cfg.arm == "asentmax":
            from entmax import entmax_bisect
            scale = self._scale(x, S)                       # (B,n_scaled,S)
            if self.n_scaled < self.h:
                # scale only the LAST n_scaled heads (_apply_length_scaling);
                # ALiBi heads keep scale 1
                ones = torch.ones(scale.shape[0], self.h - self.n_scaled, S,
                                  device=scale.device, dtype=scale.dtype)
                scale = torch.cat([ones, scale], dim=1)
            qf = q.float() * scale.unsqueeze(-1)
            scores = torch.einsum("bhsd,bhtd->bhst", qf, k.float()) * sm
            if self.cfg.nape:
                scores = scores + alibi_bias(self.slopes, S, x.device)
            causal = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, -1e9)
            p = entmax_bisect(scores, alpha=1.5, dim=-1)
            o = torch.einsum("bhst,bhtd->bhsd", p.to(v.dtype), v)
        elif self.cfg.arm == "softmaxd":
            # dense softmax, NAPE-capable — their softmax row is
            # attn_scale_type=null, so no adaptive scale here
            scores = torch.einsum("bhsd,bhtd->bhst",
                                  q.float(), k.float()) * sm
            if self.cfg.nape:
                scores = scores + alibi_bias(self.slopes, S, x.device)
            causal = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, -1e9)
            p = F.softmax(scores, dim=-1)
            o = torch.einsum("bhst,bhtd->bhsd", p.to(v.dtype), v)
        elif self.cfg.arm == "asstjd":
            # dense AS-Stieltjes: identical scaffold to the asentmax arm
            # (same scale, same NAPE, same masking) with the normalized
            # Stieltjes mapping (detached-lambda bisection = Jack's
            # a19bb33 `stieltjes` semantics, validated vs the kernel)
            scale = self._scale(x, S)
            if self.n_scaled < self.h:
                ones = torch.ones(scale.shape[0], self.h - self.n_scaled, S,
                                  device=scale.device, dtype=scale.dtype)
                scale = torch.cat([ones, scale], dim=1)
            qf = q.float() * scale.unsqueeze(-1)
            scores = torch.einsum("bhsd,bhtd->bhst", qf, k.float()) * sm
            if self.cfg.nape:
                scores = scores + alibi_bias(self.slopes, S, x.device)
            causal = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, -1e9)
            p = stieltjes_dense_probs(scores, self.cfg.stj_q)
            o = torch.einsum("bhst,bhtd->bhsd", p.to(v.dtype), v)
        else:
            raise ValueError(self.cfg.arm)
        o = o.transpose(1, 2).reshape(B, S, self.h * self.hd)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = ArmAttn(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn, bias=False), nn.GELU(),
            nn.Linear(cfg.ffn, cfg.d_model, bias=False))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MQMTARModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB, bias=False)
        self.head.weight = self.emb.weight
        # GPT-style init (default N(0,1) embedding + tied head gives init CE
        # ~475 vs ln(260)=5.6 — fatal at lr=1e-5). beta/gamma projections are
        # re-initialized afterwards to their reference default (nn.Linear
        # kaiming-uniform), matching adapt-softplus-tanh.
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if isinstance(mod, nn.Linear) and mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        for mod in self.modules():
            if isinstance(mod, ArmAttn) and hasattr(mod, "beta_proj"):
                mod.beta_proj.reset_parameters()
                mod.gamma_proj.reset_parameters()

    def forward(self, idx):
        x = self.emb(idx)                      # NoPE token path
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Eval: greedy generation, exact match
# ---------------------------------------------------------------------------

@torch.no_grad()
def diag_samples(model, src_list, trg_list, device, n):
    """Print generation vs target token-by-token for the first n samples."""
    model.eval()
    for i in range(n):
        prompt = np.concatenate(([BOS], src_list[i], [SEP]))
        x = torch.from_numpy(prompt[None].astype(np.int64)).to(device)
        for _ in range(TRG_LEN):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            x = torch.cat([x, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        gen = x[0, -TRG_LEN:].cpu().numpy()
        trg = trg_list[i][:TRG_LEN]
        marks = "".join("=" if g == t else "X" for g, t in zip(gen, trg))
        ctx = " ".join(map(str, src_list[i][:40]))
        qs = " ".join(map(str, src_list[i][-12:]))
        print(f"[diag {i}] ctx[:40]: {ctx}")
        print(f"[diag {i}] queries : {qs}")
        print(f"[diag {i}] trg     : {' '.join(map(str, trg))}")
        print(f"[diag {i}] gen     : {' '.join(map(str, gen))}")
        print(f"[diag {i}] match   : {marks}", flush=True)
    model.train()


@torch.no_grad()
def eval_split(model, src_list, trg_list, device, n_samples, bs):
    model.eval()
    n = min(n_samples, len(src_list))
    correct = 0
    for lo in range(0, n, bs):
        idxs = range(lo, min(lo + bs, n))
        prompts = [np.concatenate(([BOS], src_list[i], [SEP])) for i in idxs]
        L = max(len(p) for p in prompts)
        x = np.full((len(prompts), L), PAD, dtype=np.int64)
        for j, p in enumerate(prompts):
            x[j, L - len(p):] = p              # LEFT-pad so last pos = SEP
        x = torch.from_numpy(x).to(device)
        for _ in range(TRG_LEN):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
        gen = x[:, -TRG_LEN:].cpu().numpy()
        for j, i in enumerate(idxs):
            if np.array_equal(gen[j], trg_list[i][:TRG_LEN]):
                correct += 1
    model.train()
    return correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sdpa", "asentmax", "asstj",
                                      "softmaxd", "asstjd"],
                    required=True)
    ap.add_argument("--nape", action="store_true",
                    help="published PE: ALiBi on first H/2 heads, adaptive "
                         "scale on the NoPE half (dense arms only)")
    ap.add_argument("--stj-q", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=24414)
    ap.add_argument("--warmup", type=int, default=1250)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--diag", type=int, default=10,
                    help="print N generation-vs-target samples after eval")
    ap.add_argument("--beta-cap", type=float, default=0.0, dest="beta_cap",
                    help="bound beta = cap*sigmoid(.) instead of softplus "
                         "(0 = off; stabilizes asstjd, see header)")
    args = ap.parse_args()
    DENSE_ARMS = {"asentmax", "softmaxd", "asstjd"}
    if args.nape and args.arm not in DENSE_ARMS:
        ap.error("--nape needs a dense arm (asentmax/softmaxd/asstjd)")
    args.arm_label = (args.arm if args.arm not in ("asstj", "asstjd")
                      else f"{args.arm}-q{args.stj_q:g}") + \
                     ("-nape" if args.nape else "") + \
                     (f"-bc{args.beta_cap:g}" if args.beta_cap > 0 else "")
    args.dense = args.arm in DENSE_ARMS
    # dense-arm eval cap (N^2 fp32 scores: 16k = 17GB/sample x several
    # bisect buffers — infeasible; 8k = 4.3GB, bs=1 ok)
    max_eval_len = 8192 if args.dense else 65536

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    cfg = args
    cfg.stj_q = args.stj_q

    run = wandb.init(
        project="stieltjes-flash-attn",
        name=f"mqmtar-{args.arm_label}-{os.environ.get('SLURM_JOB_ID', 'local')}",
        config={**vars(args), "protocol": "ASEntmax MQMTAR 1/16 budget",
                "data": DATA, "port": "v2-faithful",
                "pos": "NAPE" if args.nape else "NoPE"},
    )

    print("loading data...")
    tr_src, tr_trg = read_split(os.path.join(DATA, "train"))
    splits = []
    for i, L in enumerate([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                           32768, 65536]):
        if L > max_eval_len:
            continue
        splits.append((L, *read_split(os.path.join(DATA, f"test_{i}_{L}"))))
    print(f"train={len(tr_src)}  eval splits={[s[0] for s in splits]}")

    model = MQMTARModel(cfg).to(device)
    print(f"arm={args.arm_label} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.99))

    def cosine_warmup(step):  # = transformers.get_cosine_schedule_with_warmup
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_warmup)
    rng = np.random.default_rng(args.seed)

    model.train()
    for step in range(args.steps):
        idxs = rng.integers(0, len(tr_src), size=args.bs)
        x, y, m = make_batch(tr_src, tr_trg, idxs, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits[m], y[m])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step:6d} loss {loss.item():.4f}", flush=True)
            run.log({"step": step, "train_loss": loss.item()})
        if step % 5000 == 0 and step > 0:
            acc64 = eval_split(model, splits[0][1], splits[0][2], device,
                               200, 64)
            print(f"  [val] step {step} acc@64 = {acc64:.3f}", flush=True)
            run.log({"step": step, "val_acc_64": acc64})

    ckpt = (f"/users/PAS2402/alexg/softmax/softmax-is-meh/results/"
            f"mqmtar_ckpt_{args.arm_label}_"
            f"{os.environ.get('SLURM_JOB_ID', 'local')}.pt")
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ckpt)
    print(f"checkpoint saved: {ckpt}", flush=True)

    import time
    print("\nfinal eval (exact match via greedy generation):")
    for L, s, t in splits:
        if args.dense:                 # dense O(N^2): shrink n and batch
            n = 1000 if L <= 2048 else (500 if L == 4096 else 100)
            bs = max(1, min(64, 2 ** 27 // (args.n_head * L * L)))
        else:
            # flash arms are memory-linear but the no-KV-cache generation
            # loop re-runs 13 full forwards; the Stieltjes fwd is ~10 N^2
            # sweeps (2 + num_iter), so 32k/65k at n=250 ran 1-5 HOURS
            # (jobs 12346516/829/484). n=100 at L>=32k keeps CI ~ +-8pp.
            n = 1000 if L <= 4096 else (250 if L <= 16384 else 100)
            bs = max(1, min(64, 2 ** 22 // max(L, 1)))
        t0 = time.time()
        acc = eval_split(model, s, t, device, n, bs)
        print(f"  len {L:6d}: acc {acc:.4f}  (n={n}, {time.time()-t0:.0f}s)",
              flush=True)
        run.log({f"final/acc_{L}": acc})
        run.summary[f"acc_{L}"] = acc

    if args.diag > 0:
        print("\ndiagnostic generations (test_0_64):", flush=True)
        diag_samples(model, splits[0][1], splits[0][2], device, args.diag)

    run.finish()
    print("DONE")


if __name__ == "__main__":
    main()
