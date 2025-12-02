import math
import itertools
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    import triton
    import triton.language as tl

    from src.kernel import kernel_add

    GPU_DEVICE = triton.runtime.driver.active.get_active_torch_device()
except ImportError as e:
    logging.warning(e)

from src.utils import time_it

CPU_DEVICE = "cpu"


@time_it
def cpu_simple_loop(A: list[float], B: list[float]) -> list[float]:
    yield "act"
    return [a + b for a, b in zip(A, B)]


@time_it
def cpu_with_threads(A: list[float], B: list[float]):
    """This function must be run with a freethreaded python version."""
    def add_vec(vec1: list[float], vec2: list[float]) -> list[float]:
        return [a + b for a, b in zip(vec1, vec2)]

    nb_workers = multiprocessing.cpu_count() - 1
    block_size = len(A) // nb_workers
    nb_blocks = math.ceil(len(A) / block_size)

    yield "act"

    with ThreadPoolExecutor(max_workers=nb_workers) as executor:
        futures = [
            executor.submit(add_vec,
                            A[i*block_size: (i+1)*block_size],
                            B[i*block_size: (i+1)*block_size])
            for i in range(nb_blocks)
        ]
        C = list(itertools.chain(*[fut.result() for fut in futures]))

    return C


@time_it
def cpu_pytorch(vector_size: int):
    
    A = torch.rand(vector_size, device="cpu")
    B = torch.rand(vector_size, device="cpu")
    C = A + B

    yield "act"
    return C


@time_it
def gpu_pytorch(vector_size: int):
    A = torch.rand(vector_size, device=GPU_DEVICE)
    B = torch.rand(vector_size, device=GPU_DEVICE)

    yield "act"
    return A + B


@time_it
def gpu_triton_baseline(vector_size: int):
    A = torch.rand(vector_size)
    B = torch.rand(vector_size)

    A = A.to("cuda:0")
    B = B.to("cuda:0")

    C = torch.zeros(vector_size, device="cuda:0")

    BLOCK_SIZE = 1024
    yield "act"

    grid = (math.ceil(vector_size / BLOCK_SIZE),)
    kernel_add[grid](A, B, C, vector_size, block_size=BLOCK_SIZE)

    return C


@time_it
def gpu_triton_with_copy(vector_size: int):
    A = torch.rand(vector_size, device="cpu")
    B = torch.rand(vector_size, device="cpu")
    BLOCK_SIZE = 1024

    yield "act"

    A = A.to("cuda:0")
    B = B.to("cuda:0")
    C = torch.zeros(vector_size, device="cuda:0")

    grid = (math.ceil(vector_size / BLOCK_SIZE),)
    kernel_add[grid](A, B, C, vector_size, block_size=BLOCK_SIZE)

    return C