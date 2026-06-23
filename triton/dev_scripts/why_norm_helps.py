"""
WHY does normalization help OOD? (stieltjes vs stieltjes_old, architecture_new)

Observed: the normalized `stieltjes` (probs/probs.sum()) beats the unnormalized
`stieltjes_old` by ~10pp (q=2) / ~5pp (q=4) at L=256 OOD, despite the forward
weight-sum S being within 1e-4 of 1 at all measured lengths. Two hypotheses:

H1 (forward/eval): at OOD lengths the unnormalized model's output is scaled by
   S(T); if S drifts with T, accuracy drops and renorm fixes it. (Measured S
   ~1.0001 so this should be negligible — but it's a near-free direct test.)

H2 (backward/training): the normalized mapping has a different GRADIENT.
   With lambda detached and max-centering differentiable:
     unnormalized:  dS_k = q[ r_k dP_k − [k==argmax] A ]
     normalized:    dS_k = (q/S)[ r_k (dP_k − B) − [k==argmax](A − R·B) ]
   where r=(λ−x)^{-q-1}, R=Σr, dP_j=dO·v_j, A=Σ dP·r, B=Σ dP·p = dO·O.
   The difference is a zero-sum coupling term −B·(r_k − [k==argmax] R):
   the normalized Jacobian has rows summing to 0 (shift-invariance in dP,
   like softmax), the unnormalized one does not. Training may find a
   different, better-generalizing solution. (Precedent: BS-vs-NR, where the
   entire 4pp OOD gap was the backward.)

Factorization (all PyTorch, Jack's architecture_new MLP MaxRetrievalModel,
identical init/data per seed):
  1. UNNORM      = stieltjes_old           (w;        unnorm fwd, unnorm bwd)
  2. NORM        = stieltjes               (w/S;      norm fwd,   norm bwd)
  3. NF_UB       = w / S.detach()          (norm fwd, UNNORM-style bwd: kills
                                            the coupling term, keeps 1/S scale)
  4. UF_NB       = w.detach() + (p − p.detach())
                                           (unnorm fwd, NORM bwd exactly)
Plus eval-time swaps on the trained UNNORM/NORM models (renorm on/off at eval
only) and trained-model diagnostics (S vs T, attention sharpness vs T, score
scale vs T).

Decision table:
  NF_UB≈UNNORM and UF_NB≈NORM  -> backward (gradient coupling) drives the gap
  NF_UB≈NORM  and UF_NB≈UNNORM -> forward drives it
  eval-swap moves accuracy      -> H1 (eval-time scale) contributes

Built-in --verify-only mode checks (fp64): value/Jacobian identities of every
variant against Jack's actual mapping classes before any training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

WORKTREE = "/users/PAS2402/alexg/softmax/psm-architecture-new"
sys.path.insert(0, WORKTREE)

from max_retrieval_architecture.architecture import MaxRetrievalModel  # noqa: E402
from mappings.base_cls import ProbabilitySimplexMapping  # noqa: E402
from mappings.stieltjes import StieltjesTransform as JackNorm  # noqa: E402
from mappings.stieltjes_old import StieltjesTransform as JackUnnorm  # noqa: E402


# ---------------------------------------------------------------------------
# Mappings: shared bisection core + 4 variants
# ---------------------------------------------------------------------------

def _bisect_lambda(shifted, dim, q, num_iter, eps):
    """Verbatim Jack's bisection (stieltjes / stieltjes_old agree on this)."""
    n = shifted.shape[dim]
    ref = shifted.max(dim=dim, keepdim=True).values  # shape helper (values are 0)
    lb = torch.full_like(ref, eps)
    ub = torch.full_like(ref, n ** (1.0 / q))
    for _ in range(num_iter):
        mid = (lb + ub) * 0.5
        f = torch.sum(
            torch.pow((mid - shifted).clamp(min=eps), -q), dim=dim, keepdim=True
        ) - 1.0
        lb = torch.where(f > 0.0, mid, lb)
        ub = torch.where(f <= 0.0, mid, ub)
    return (lb + ub) * 0.5


class _Base(ProbabilitySimplexMapping):
    def __init__(self, q: float = 1.0, num_iter: int = 16, eps: float = 1e-9):
        super().__init__()
        self._q = q
        self._num_iter = num_iter
        self._eps = eps

    def _core(self, logits, dim):
        x_max = logits.max(dim=dim, keepdim=True).values  # differentiable (argmax term)
        shifted = logits - x_max
        lam = _bisect_lambda(shifted, dim, self._q, self._num_iter, self._eps)
        w = torch.pow((lam - shifted).clamp(min=self._eps), -self._q)
        S = w.sum(dim=dim, keepdim=True).clamp(min=self._eps)
        return w, S


class MapUnnorm(_Base):
    """== stieltjes_old: returns w, unnormalized."""
    def translate_logits(self, logits, dim, **kw):
        w, _S = self._core(logits, dim)
        return w


class MapNorm(_Base):
    """== stieltjes: returns w / S."""
    def translate_logits(self, logits, dim, **kw):
        w, S = self._core(logits, dim)
        return w / S


class MapNormFwdUnnormBwd(_Base):
    """Forward = w/S; backward treats S as a constant (no coupling term)."""
    def translate_logits(self, logits, dim, **kw):
        w, S = self._core(logits, dim)
        return w / S.detach()


class MapUnnormFwdNormBwd(_Base):
    """Forward = w; backward = exactly the normalized gradient dp/ds."""
    def translate_logits(self, logits, dim, **kw):
        w, S = self._core(logits, dim)
        p = w / S
        return w.detach() + (p - p.detach())


class _MapEnum:
    """Duck-type SimplexMappingEnum: architecture accesses .value (a class)."""
    def __init__(self, cls):
        self.value = cls


VARIANTS = [
    ("UNNORM", MapUnnorm),
    ("NORM", MapNorm),
    ("NF_UB", MapNormFwdUnnormBwd),
    ("UF_NB", MapUnnormFwdNormBwd),
]


# ---------------------------------------------------------------------------
# Verification (fp64): value + Jacobian identities vs Jack's classes
# ---------------------------------------------------------------------------

def verify(device):
    print("=" * 76)
    print("VERIFY (fp64): variant identities vs Jack's mappings")
    print("=" * 76)
    torch.manual_seed(0)
    ok = True
    for q in [2.0, 4.0]:
        for T in [16, 64]:
            logits = (torch.randn(4, 1, T, dtype=torch.float64, device=device) * 1.5)

            jack_u = JackUnnorm(q=q)
            jack_n = JackNorm(q=q)
            mine = {name: cls(q=q).to(device).double() for name, cls in VARIANTS}

            vu = mine["UNNORM"].translate_logits(logits, -1)
            vn = mine["NORM"].translate_logits(logits, -1)
            e1 = (vu - jack_u.translate_logits(logits, -1)).abs().max().item()
            e2 = (vn - jack_n.translate_logits(logits, -1)).abs().max().item()

            # value identities of hybrids
            e3 = (mine["NF_UB"].translate_logits(logits, -1) - vn).abs().max().item()
            e4 = (mine["UF_NB"].translate_logits(logits, -1) - vu).abs().max().item()

            # Jacobian identities on a single row
            x = logits[0, 0].clone()

            def J(map_obj):
                def f(z):
                    return map_obj.translate_logits(z.unsqueeze(0), -1).squeeze(0)
                return torch.autograd.functional.jacobian(f, x)

            Ju, Jn = J(mine["UNNORM"]), J(mine["NORM"])
            Jnfub, Jufnb = J(mine["NF_UB"]), J(mine["UF_NB"])
            w, S = mine["UNNORM"]._core(x.unsqueeze(0), -1)
            Sv = S.squeeze().item()
            e5 = (Jufnb - Jn).abs().max().item()          # UF_NB bwd == NORM bwd
            e6 = (Jnfub * Sv - Ju).abs().max().item()     # NF_UB bwd == UNNORM bwd / S

            line = (f"  q={q:>3.0f} T={T:>3}: |U-jack|={e1:.1e} |N-jack|={e2:.1e} "
                    f"|NF_UB-val-N|={e3:.1e} |UF_NB-val-U|={e4:.1e} "
                    f"|J(UF_NB)-J(N)|={e5:.1e} |S*J(NF_UB)-J(U)|={e6:.1e}")
            bad = max(e1, e2, e3, e4, e5, e6) > 1e-10
            ok = ok and not bad
            print(line + ("   <-- FAIL" if bad else ""))
    print(f"  >>> {'ALL IDENTITIES HOLD' if ok else 'IDENTITY FAILURE — DO NOT TRAIN'}")
    return ok


# ---------------------------------------------------------------------------
# Data / train / eval (identical to validated architecture_new comparisons)
# ---------------------------------------------------------------------------

def _set_seeds(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_batch(bs, T, n_classes, device, gen):
    pri = torch.rand(bs, T, device=device, generator=gen)
    cls = torch.randint(0, n_classes, (bs, T), device=device, generator=gen)
    tgt = cls.gather(1, pri.argmax(1, keepdim=True)).squeeze(1).long()
    items = torch.cat([pri.unsqueeze(-1), F.one_hot(cls, n_classes).float()], -1)
    qq = torch.rand(bs, 1, device=device, generator=gen)
    return items, qq, tgt


def train(model, *, seq_len, n_classes, device, steps, bs, lr, wd, warmup, seed):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                        pct_start=warmup / steps, anneal_strategy="cos")
    lf = nn.CrossEntropyLoss()
    for step in tqdm(range(steps), desc="train", leave=False):
        g = torch.Generator(device=device)
        g.manual_seed(seed * 1_000_003 + step)
        items, qq, tgt = make_batch(bs, seq_len, n_classes, device, g)
        opt.zero_grad(set_to_none=True)
        loss = lf(model(items, qq), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()


@torch.no_grad()
def evaluate(model, *, seq_len, n_classes, device, samples, bs, seed=12345):
    model.eval()
    correct = total = 0
    ebs = bs * 2
    for i in range(int(np.ceil(samples / ebs))):
        g = torch.Generator(device=device)
        g.manual_seed(seed + i)
        items, qq, tgt = make_batch(ebs, seq_len, n_classes, device, g)
        correct += (model(items, qq).argmax(-1) == tgt).sum().item()
        total += tgt.numel()
    return 100.0 * correct / max(total, 1)


def build_model(cls, device, *, d_emb, n_classes, item_dim, sq, seed):
    _set_seeds(seed)
    return MaxRetrievalModel(
        simplex_mapping=_MapEnum(cls), d_emb=d_emb, n_classes=n_classes,
        item_input_dim=item_dim, query_input_dim=1,
        attn_score_scale="inv_sqrt_d", q=sq,
    ).to(device)


# ---------------------------------------------------------------------------
# Trained-model diagnostics: S, sharpness, score scale vs length
# ---------------------------------------------------------------------------

@torch.no_grad()
def diagnose(model, sq, *, lengths, n_classes, device, bs=256):
    out = {}
    for L in lengths:
        g = torch.Generator(device=device)
        g.manual_seed(99)
        items, qq, _ = make_batch(bs, L, n_classes, device, g)
        h_items = model.psi_x(items)
        h_query = model.psi_q(qq.unsqueeze(-1))
        q = model.q_proj(h_query)
        k = model.k_proj(h_items)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (model.d_emb ** -0.5)
        x_max = scores.max(-1, keepdim=True).values
        shifted = scores - x_max
        lam = _bisect_lambda(shifted, -1, sq, 16, 1e-9)
        w = (lam - shifted).clamp(min=1e-9).pow(-sq)
        S = w.sum(-1, keepdim=True)
        p = w / S
        out[L] = {
            "S_mean": S.mean().item(),
            "S_absdev_max": (S - 1).abs().max().item(),
            "pmax_mean": p.max(-1).values.mean().item(),
            "entropy_mean": (-(p.clamp(min=1e-12) * p.clamp(min=1e-12).log())
                             .sum(-1).mean().item()),
            "score_max_mean": scores.max(-1).values.mean().item(),
            "score_std_mean": scores.std(-1).mean().item(),
            "lam_mean": lam.mean().item(),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--id-len", type=int, default=16)
    ap.add_argument("--ood-lens", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not verify(device):
        sys.exit(1)
    if args.verify_only:
        return

    n_classes, d_emb = 10, 128
    item_dim = 1 + n_classes
    lengths = [args.id_len] + args.ood_lens

    acc = {}   # acc[q][entry][L] -> list over seeds
    diag = {}

    for sq in args.qs:
        acc[sq] = {}
        diag[sq] = {}
        for seed in args.seeds:
            kept = {}
            for name, cls in VARIANTS:
                model = build_model(cls, device, d_emb=d_emb, n_classes=n_classes,
                                    item_dim=item_dim, sq=sq, seed=seed)
                train(model, seq_len=args.id_len, n_classes=n_classes, device=device,
                      steps=args.steps, bs=256, lr=1e-3, wd=1e-4,
                      warmup=max(1, args.steps // 10), seed=seed)
                accs = {L: evaluate(model, seq_len=L, n_classes=n_classes,
                                    device=device,
                                    samples=2048 if L == args.id_len else 1024, bs=256)
                        for L in lengths}
                acc[sq].setdefault(name, {L: [] for L in lengths})
                for L in lengths:
                    acc[sq][name][L].append(accs[L])
                if name in ("UNNORM", "NORM"):
                    kept[name] = model
                print(f"  q={sq} seed={seed} {name:6s}: " +
                      " ".join(f"L{L}={accs[L]:.1f}" for L in lengths))

            # ----- eval-time swaps (H1) on the kept trained models -----
            swaps = [
                ("UNNORM+evalRenorm", kept["UNNORM"], MapNorm),
                ("NORM+evalNoRenorm", kept["NORM"], MapUnnorm),
            ]
            for sname, model, swap_cls in swaps:
                orig = model._translate_logits
                model._translate_logits = swap_cls(q=sq).to(device)
                accs = {L: evaluate(model, seq_len=L, n_classes=n_classes,
                                    device=device,
                                    samples=2048 if L == args.id_len else 1024, bs=256)
                        for L in lengths}
                model._translate_logits = orig
                acc[sq].setdefault(sname, {L: [] for L in lengths})
                for L in lengths:
                    acc[sq][sname][L].append(accs[L])
                print(f"  q={sq} seed={seed} {sname:18s}: " +
                      " ".join(f"L{L}={accs[L]:.1f}" for L in lengths))

            # ----- diagnostics on seed-0 trained models -----
            if seed == args.seeds[0]:
                for name in ("UNNORM", "NORM"):
                    diag[sq][name] = diagnose(kept[name], sq,
                                              lengths=[16, 256, 1024],
                                              n_classes=n_classes, device=device)

        # ----- per-q summary table -----
        print(f"\n===== SUMMARY q={sq} (mean ± std over seeds {args.seeds}) =====")
        entries = ["UNNORM", "NORM", "NF_UB", "UF_NB",
                   "UNNORM+evalRenorm", "NORM+evalNoRenorm"]
        header = f"{'entry':>18} | " + " ".join(f"{('L'+str(L)):>11}" for L in lengths)
        print(header)
        print("-" * len(header))
        for name in entries:
            row = []
            for L in lengths:
                v = np.array(acc[sq][name][L])
                row.append(f"{v.mean():6.2f}±{v.std():4.2f}")
            print(f"{name:>18} | " + " ".join(f"{c:>11}" for c in row))

        print(f"\n  Diagnostics (seed {args.seeds[0]} trained models):")
        for name in ("UNNORM", "NORM"):
            for L, d in diag[sq][name].items():
                print(f"    {name:6s} L={L:5d}: S={d['S_mean']:.6f} "
                      f"|S-1|max={d['S_absdev_max']:.2e} pmax={d['pmax_mean']:.3f} "
                      f"H={d['entropy_mean']:.3f} smax={d['score_max_mean']:.2f} "
                      f"sstd={d['score_std_mean']:.3f} lam={d['lam_mean']:.4f}")

    out = {
        "qs": args.qs, "seeds": args.seeds, "steps": args.steps,
        "lengths": lengths,
        "acc": {str(q): {n: {str(L): v for L, v in d.items()}
                         for n, d in acc[q].items()} for q in acc},
        "diag": {str(q): diag[q] for q in diag},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
