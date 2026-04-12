import importlib
import torch
import triton

from fused_att import attention, is_blackwell, is_hopper, DEVICE

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 4, 32, 64


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N_CTX'],
        x_vals=[2**i for i in range(8, 15)],
        line_arg='provider',
        line_vals=['triton-fp16'] + (['triton-fp8'] if TORCH_HAS_FP8 else []) +
        (['flash'] if HAS_FLASH else []),
        line_names=['Triton [FP16]'] + (['Triton [FP8]'] if TORCH_HAS_FP8 else []) +
        (['Flash-2'] if HAS_FLASH else []),
        styles=[('red', '-'), ('blue', '-'), ('green', '-')],
        ylabel='TFLOPS',
        plot_name='fused-attention-performance',
        args={
            'H': N_HEADS,
            'BATCH': BATCH,
            'HEAD_DIM': HEAD_DIM,
            'mode': 'fwd',
            'causal': False,
            'warp_specialize': False,
        },
    ))
def benchmark(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, warp_specialize)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    benchmark.run(save_path='.', print_data=True)
