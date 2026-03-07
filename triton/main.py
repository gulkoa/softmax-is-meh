import triton
import triton.language as tl

import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()
print(f"device is {DEVICE}")


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)
    
    
def add(
    x,
    y
):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    
    n_elements = output.numel()
    # print(f"{=n_elements}")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

def main():
    torch.manual_seed(42)
    size = 98432
    x, y = torch.rand(size, device=DEVICE), torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    
    diff = output_triton - output_torch
    print("Maximum absolute diff is", torch.max(torch.abs(diff)).item())

if __name__ == "__main__":
    main()
