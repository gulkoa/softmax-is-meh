"""
Isolate the FLASH FORWARD against architecture_new's `stieltjes` reference.

architecture_new (Jack's MLP MaxRetrievalModel) computes single-query attention:
  q=(B,1,d), k,v=(B,T,d) -> scores=(B,1,T)*scale -> weights=stieltjes(scores)
  -> z = weights @ v -> phi(z).

The new `stieltjes` mapping bisects for lambda AND THEN EXPLICITLY RENORMALIZES:
  probs = (lam - shifted)^(-q);  return probs / probs.sum()    <-- normalization

The Triton flash forward does NOT renormalize (it relies on the NR solve giving
sum ~= 1). Two candidate forward discrepancies:
  (A) BUG 2: padded key columns at T=16 (T < BLOCK_N=64). [fixed in kernel]
  (B) missing renormalization: at high q / few iters the NR weight-sum != 1,
      so flash z is systematically scaled vs the renormalized reference.

This script, on architecture_new q/k/v at T=16, compares the attention output z:
  z_ref     = stieltjes(scores) @ v                  (normalized reference)
  z_flash   = Triton flash forward, take row 0       (no renorm)
  z_flashN  = flash weights / flash weights.sum() @ v  (flash + explicit renorm)
and reports the flash weight-row-sum (should be 1 if converged) and lambda.

Sweeps q in {2,4,8,16,32,64} (architecture_new's values) and num_iter, at T=16
and a few OOD lengths.
"""
import sys
import torch
import triton

WORKTREE = "/users/PAS2402/alexg/softmax/psm-architecture-new"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")

from max_retrieval_architecture.architecture import MaxRetrievalModel  # noqa: E402
from mappings.type_enum import SimplexMappingEnum  # noqa: E402
from mappings.stieltjes import StieltjesTransform as StieltjesNew  # noqa: E402
import stieltjes_flash_attn as sfa  # noqa: E402


def flash_fwd_full(q_t, k_t, v_t, sm_scale, sq, num_iter):
    """Run the raw Triton forward; return (O, lambda_abs) for q_t/k_t/v_t of
    shape (B,H,N,D)."""
    B, H, N, D = q_t.shape
    o = torch.empty_like(q_t)
    lam = torch.empty((B * H, N), device=q_t.device, dtype=torch.float32)
    d_sum = torch.empty((B * H, N), device=q_t.device, dtype=torch.float32)
    argmax = torch.empty((B * H, N), device=q_t.device, dtype=torch.int32)
    lambda_init = torch.full((N,), 1.1, device=q_t.device, dtype=torch.float32)
    if D <= 64:
        BLOCK_M, BLOCK_N = 128, 64
    elif q_t.element_size() >= 4:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    sfa._stieltjes_attn_fwd[grid](
        q_t, k_t, v_t, o, lam, d_sum, argmax, lambda_init,
        q_t.stride(0), q_t.stride(1), q_t.stride(2), q_t.stride(3),
        k_t.stride(0), k_t.stride(1), k_t.stride(2), k_t.stride(3),
        v_t.stride(0), v_t.stride(1), v_t.stride(2), v_t.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale, N, sq=sq, NUM_ITER=num_iter, EPS=1e-6,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=False,
    )
    return o, lam  # lam is absolute lambda (lambd + row_max) per row


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    torch.manual_seed(0)

    d_emb, n_classes = 128, 10
    item_dim = 1 + n_classes
    sm_scale = d_emb ** -0.5

    # Build architecture_new model (MLP) to get realistic projections.
    model = MaxRetrievalModel(
        simplex_mapping=SimplexMappingEnum.stieltjes, d_emb=d_emb,
        n_classes=n_classes, item_input_dim=item_dim, query_input_dim=1,
        attn_score_scale="inv_sqrt_d", q=4.0,
    ).to(device)
    model.eval()

    def sample(T, bs=64):
        pri = torch.rand(bs, T, device=device)
        cls = torch.randint(0, n_classes, (bs, T), device=device)
        items = torch.cat([pri.unsqueeze(-1),
                           torch.nn.functional.one_hot(cls, n_classes).float()], -1)
        queries = torch.rand(bs, 1, device=device)
        return items, queries

    print("Comparing attention output z three ways, on architecture_new q/k/v.")
    print("z_ref = normalized stieltjes; z_flash = Triton fwd (no renorm);")
    print("z_flashN = flash weights renormalized. Also flash weight-row-sum.\n")

    for T in [16, 64, 128]:
        items, queries = sample(T)
        with torch.no_grad():
            h_items = model.psi_x(items)
            h_query = model.psi_q(queries.unsqueeze(-1))
            q = model.q_proj(h_query)        # (B,1,d)
            k = model.k_proj(h_items)        # (B,T,d)
            v = model.v_proj(h_items)        # (B,T,d)
            scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale  # (B,1,T)

        print(f"===== T={T} =====")
        print(f"{'q':>4} {'iter':>5} | {'z|ref-flash|':>13} {'z|ref-flashN|':>14} "
              f"{'flash_sum(mean)':>16} {'w|ref-flashN|':>13}")
        for sq in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
            ref_map = StieltjesNew(q=sq, num_iter=30, eps=1e-9)
            w_ref = ref_map.translate_logits(scores, dim=-1)      # (B,1,T) normalized
            z_ref = torch.matmul(w_ref, v).squeeze(1)             # (B,d)

            for num_iter in [10, 30]:
                B = scores.shape[0]
                # Tile single query to T rows; (B,1,T,d)
                q_t = q.expand(B, T, d_emb).unsqueeze(1).contiguous()
                k_t = k.unsqueeze(1).contiguous()
                v_t = v.unsqueeze(1).contiguous()
                o, lam = flash_fwd_full(q_t, k_t, v_t, sm_scale, sq, num_iter)
                z_flash = o[:, 0, 0, :]                            # (B,d)

                # Recompute flash weights from the flash lambda to inspect sum
                lam_row = lam.view(B, 1, T)[:, :, 0:1]             # (B,1,1) lambda for query row 0
                # absolute lambda; weights = (lam - scores)^(-q)
                diff = (lam_row - scores).clamp(min=1e-6)          # (B,1,T)
                w_flash = diff.pow(-sq)                            # (B,1,T)  un-normalized
                flash_sum = w_flash.sum(dim=-1)                   # (B,1)
                w_flashN = w_flash / flash_sum.unsqueeze(-1).clamp(min=1e-9)
                z_flashN = torch.matmul(w_flashN, v).squeeze(1)

                e_rf = (z_ref - z_flash).abs().max().item()
                e_rfn = (z_ref - z_flashN).abs().max().item()
                e_wrfn = (w_ref - w_flashN).abs().max().item()
                print(f"{sq:>4.0f} {num_iter:>5} | {e_rf:>13.3e} {e_rfn:>14.3e} "
                      f"{flash_sum.mean().item():>16.5f} {e_wrfn:>13.3e}")
        print()


if __name__ == "__main__":
    main()
