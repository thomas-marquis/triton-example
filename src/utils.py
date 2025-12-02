import random
import inspect
from time import perf_counter
from typing import Generator, Callable, Any
from functools import wraps


def generate_rand_vec(size: int) -> list[float]:
    return [random.random() for _ in range(size)]


def time_it[T](func: Callable[..., Generator[str, None, T]]) -> Callable[..., tuple[T, float]]:
    assert inspect.isgeneratorfunction(func), "generator function expected"

    @wraps(func)
    def wrapper(*args, **kwargs) -> tuple[T, float]:
        gen = func(*args, **kwargs)
        t0 = None
        elapsed = None
        res = None

        while True:
            try:
                match next(gen):
                    case "arrange":
                        continue
                    case "act":
                        t0 = perf_counter()
                    case "cleanup":
                        elapsed = perf_counter() - t0
            except StopIteration as e:
                if t0 is None:
                    raise ValueError("yield 'act' expected")
                if elapsed is None:
                    elapsed = perf_counter() - t0
                res = e.value
                break

        if elapsed is not None:
            print(f"elapsed time for {func.__name__}: {elapsed:.5f}s")
        return res, elapsed

    return wrapper