import triton
import triton.language as tl

@triton.jit
def kernel_add(A_ptr, B_ptr, C_ptr, n_elements, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)

    c = a + b

    tl.store(C_ptr + offsets, c, mask=mask)



