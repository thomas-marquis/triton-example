import triton
import triton.language as tl

@triton.jit
def kernel_add_(A, B, C, n_elements, block_size):
    pid = tl.program_id(axis=0)

    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    x = tl.load(A + offsets, mask=mask)
    y = tl.load(B + offsets, mask=mask)

    output = x + y

    tl.store(C + offsets, output, mask=mask)
