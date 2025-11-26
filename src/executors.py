import torch
import triton

from src.utils import time_it, generate_rand_vec
from src.kernel import kernel_add

VEC_SIZE = 50_000_000
GPU_DEVICE = triton.runtime.driver.active.get_active_torch_device()
CPU_DEVICE = "cpu"


@time_it
def simple_loop():
    A = generate_rand_vec(VEC_SIZE)
    B = generate_rand_vec(VEC_SIZE)

    yield "act"
    return [a + b for a, b in zip(A, B)]


@time_it
def pytorch_cpu():
    A = torch.rand(VEC_SIZE, device=CPU_DEVICE)
    B = torch.rand(VEC_SIZE, device=CPU_DEVICE)

    yield "act"
    return A + B


@time_it
def pytorch_gpu():
    A = torch.rand(VEC_SIZE, device=GPU_DEVICE)
    B = torch.rand(VEC_SIZE, device=GPU_DEVICE)

    yield "act"
    return A + B


@time_it
def triton_baseline():
    A = torch.rand(VEC_SIZE, device=GPU_DEVICE)
    B = torch.rand(VEC_SIZE, device=GPU_DEVICE)
    output = torch.empty_like(A)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    yield "act"
    kernel_add[grid](A, B, output, n_elements, BLOCK_SIZE=1)

    return output


@time_it
def faster_triton():
    A = torch.rand(VEC_SIZE, device=GPU_DEVICE)
    B = torch.rand(VEC_SIZE, device=GPU_DEVICE)
    output = torch.empty_like(A)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    yield "act"
    kernel_add[grid](A, B, output, n_elements, BLOCK_SIZE=1024)

    return output
