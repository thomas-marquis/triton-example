import triton
import triton.language as tl



@triton.jit
def a_triton_kernel(A, B, C, nb_elements, block_size):
    ...


@triton.jit
def kernel_add(
    x_ptr: tl.pointer_type,
    y_ptr: tl.pointer_type,
    output_ptr: tl.pointer_type,
    n_elements,
    BATCH_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    block_start = pid * BATCH_SIZE
    offsets = block_start + tl.arange(0, BATCH_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    if pid == 0 or True:
        # tl.device_print("=============== pid", pid)
        # tl.device_print("offsets", offsets)
        # tl.device_print("numprogs", tl.num_programs(axis=0))
        # tl.device_print("output", output)
        # tl.device_print("BATCH_SIZE", BATCH_SIZE)
        # tl.device_print("n_elements", n_elements)
        # tl.device_print("x", x)
        # tl.device_print("y", y)
        tl.device_print("mask", mask)

    tl.store(output_ptr + offsets, output, mask=mask)
