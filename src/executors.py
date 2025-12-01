import math
import itertools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import torch
import triton

import triton.language as tl
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
def with_threads():
    # A = generate_rand_vec(VEC_SIZE)
    # B = generate_rand_vec(VEC_SIZE)
    A = list(range(0, 100, 10))
    B = list(range(0, 1000, 100))

    yield "act"

    def add_vec(vec1: list[float], vec2: list[float]) -> list[float]:
        return [a + b for a, b in zip(vec1, vec2)]

    nb_cpu = multiprocessing.cpu_count()
    block_size = 4
    nb_blocks = math.ceil(len(A) / block_size)
    with ThreadPoolExecutor(max_workers=nb_cpu) as executor:
        futures = [
            executor.submit(add_vec,
                            A[i*block_size: (i+1)*block_size],
                            B[i*block_size: (i+1)*block_size])
            for i in range(nb_blocks)
        ]
        C = list(itertools.chain(*[fut.result() for fut in futures]))

    return C


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
    # A = torch.rand(VEC_SIZE, device=GPU_DEVICE)
    # B = torch.rand(VEC_SIZE, device=GPU_DEVICE)
    n_elements = 1000
    A = torch.arange(0, 1000, 10, device=GPU_DEVICE)
    B = torch.arange(0, 10000, 100, device=GPU_DEVICE)
    output = torch.zeros(n_elements, device=GPU_DEVICE)

    # grid = lambda meta: (triton.cdiv(n_elements, meta['BATCH_SIZE']),)
    yield "act"
    kernel_add[(8,)](A, B, output, n_elements, BATCH_SIZE=128)

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
