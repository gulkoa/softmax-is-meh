"""
Decompose the forward error into three candidate sources:
  (1) BUG 2: padded KEY columns at N not a multiple of BLOCK_N=64 (non-causal).
  (2) num_iter mismatch: the self-test compares Triton(num_iter=5) to ref(10).
  (3) fp16 precision.

PART A — fp32, MATCHED num_iter=20: isolates BUG 2 only. Predict tiny error at
         N multiple of 64, large error at non-multiples (non-causal).
PART B — fp16, MATCHED num_iter=20: adds fp16 effect on top of (1).
PART C — num_iter sweep at N=128 (multiple of 64), fp16: does Triton converge to
         ref as iters increase? Isolates (2).
"""
import sys
import torch

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
from stieltjes_flash_attn import stieltjes_attention, stieltjes_attention_ref


def fwd_err(N, D, q, causal, dtype, n_iter_tri, n_iter_ref, device, seed=0):
    B, H = 2, 4
    sm_scale = 1.0 / (D ** 0.5)
    torch.manual_seed(seed)
    qd = torch.randn(B, H, N, D, device=device, dtype=dtype)
    kd = torch.randn(B, H, N, D, device=device, dtype=dtype)
    vd = torch.randn(B, H, N, D, device=device, dtype=dtype)
    ref = stieltjes_attention_ref(qd.float(), kd.float(), vd.float(), sm_scale,
                                  causal=causal, stieltjes_q=q,
                                  num_iter=n_iter_ref, eps=1e-6)
    tri = stieltjes_attention(qd, kd, vd, causal=causal, sm_scale=sm_scale,
                              stieltjes_q=q, num_iter=n_iter_tri).float()
    e = (tri - ref).abs()
    return e.max().item(), e.mean().item()


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    D = 64

    print("=" * 80)
    print("PART A — fp32, MATCHED num_iter=20 (isolates BUG 2: padded key columns)")
    print("=" * 80)
    print(f"{'N':>5} {'mult64?':>8} {'causal':>7} {'q':>4} | {'max_err':>11} {'mean_err':>11}")
    print("-" * 80)
    for causal in [False, True]:
        for N in [32, 64, 96, 128, 160, 256]:
            for q in [1.0, 4.0]:
                mx, mn = fwd_err(N, D, q, causal, torch.float32, 20, 20, device)
                print(f"{N:>5} {str(N%64==0):>8} {str(causal):>7} {q:>4.0f} | "
                      f"{mx:>11.3e} {mn:>11.3e}")
        print()

    print("=" * 80)
    print("PART B — fp16, MATCHED num_iter=20 (BUG 2 + fp16)")
    print("=" * 80)
    print(f"{'N':>5} {'mult64?':>8} {'causal':>7} {'q':>4} | {'max_err':>11} {'mean_err':>11}")
    print("-" * 80)
    for causal in [False, True]:
        for N in [64, 96, 128, 256]:
            for q in [1.0, 4.0]:
                mx, mn = fwd_err(N, D, q, causal, torch.float16, 20, 20, device)
                print(f"{N:>5} {str(N%64==0):>8} {str(causal):>7} {q:>4.0f} | "
                      f"{mx:>11.3e} {mn:>11.3e}")
        print()

    print("=" * 80)
    print("PART C — num_iter sweep at N=128 (mult of 64), non-causal, fp16")
    print("  Triton num_iter varies; ref fixed at 20. Isolates num_iter effect.")
    print("=" * 80)
    print(f"{'q':>4} {'tri_iter':>9} | {'max_err':>11} {'mean_err':>11}")
    print("-" * 50)
    for q in [1.0, 4.0]:
        for ni in [3, 5, 10, 20, 30]:
            mx, mn = fwd_err(128, D, q, False, torch.float16, ni, 20, device)
            print(f"{q:>4.0f} {ni:>9} | {mx:>11.3e} {mn:>11.3e}")
        print()


if __name__ == "__main__":
    main()
