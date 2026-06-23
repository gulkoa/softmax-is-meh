"""
Truth table on the REAL Triton kernel to confirm the NaN root cause.

Hypothesis: NaN in dV/dK comes from PADDED query rows when N_CTX is not a
multiple of BLOCK_M (=128 for D<=64). Padded rows load lam_row=0 -> diff=EPS
-> weights=EPS^-q -> overflow -> inf*0=NaN in the dV/dK matmuls.

Critical falsifiable prediction:
  - N in {128, 256} (exact multiples of BLOCK_M=128): NO padded rows -> NO NaN
    at any dtype, any q.
  - N in {32, 64} (padding): NaN per the dtype/q overflow pattern:
      fp16: dV NaN at all q; dK NaN only at q=16
      fp32: dV/dK NaN only at q=16

If N=128/256 show NaN, the padding hypothesis is WRONG.

We use B=1, H=1, D=64 (simplest), random normal inputs.
"""
import sys
import torch

sys.path.insert(0, "/users/PAS2402/alexg/softmax/softmax-is-meh/triton")
from stieltjes_flash_attn import stieltjes_attention  # noqa: E402


def check(N, dtype, q, block_lambda_grad):
    torch.manual_seed(0)
    B, H, D = 1, 1, 64
    sm_scale = 1.0 / (D ** 0.5)
    qd = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    kd = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    vd = torch.randn(B, H, N, D, device="cuda", dtype=dtype, requires_grad=True)
    o = stieltjes_attention(qd, kd, vd, causal=False, sm_scale=sm_scale,
                            stieltjes_q=q, num_iter=10,
                            block_lambda_grad=block_lambda_grad)
    fwd_nan = torch.isnan(o).any().item()
    do = torch.randn_like(o)
    o.backward(do)
    return {
        "fwd_O": fwd_nan,
        "dQ": torch.isnan(qd.grad).any().item(),
        "dK": torch.isnan(kd.grad).any().item(),
        "dV": torch.isnan(vd.grad).any().item(),
    }


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("D=64 -> BLOCK_M=128, BLOCK_N=64")
    print("Padding occurs when N is NOT a multiple of 128.\n")

    for block_lambda_grad in [False, True]:
        mode = "BS-mode (block_lambda_grad=True)" if block_lambda_grad else "IFT-mode (default)"
        print("=" * 84)
        print(f"  {mode}")
        print("=" * 84)
        print(f"{'N':>5} {'pad?':>5} {'dtype':>6} {'q':>4} | "
              f"{'fwdO_nan':>9} {'dQ_nan':>7} {'dK_nan':>7} {'dV_nan':>7}")
        print("-" * 84)
        for N in [32, 64, 128, 256]:
            padded = (N % 128) != 0
            for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
                for q in [1.0, 4.0, 16.0]:
                    try:
                        r = check(N, dtype, q, block_lambda_grad)
                        print(f"{N:>5} {str(padded):>5} {dtype_name:>6} {q:>4.0f} | "
                              f"{str(r['fwd_O']):>9} {str(r['dQ']):>7} "
                              f"{str(r['dK']):>7} {str(r['dV']):>7}")
                    except Exception as e:
                        print(f"{N:>5} {str(padded):>5} {dtype_name:>6} {q:>4.0f} | ERROR: {e}")
            print()


if __name__ == "__main__":
    main()
