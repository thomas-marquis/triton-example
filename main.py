from src import executors
from src.utils import generate_rand_vec

VEC_SIZE = 10_000_000


def main() -> None:
    # A = generate_rand_vec(VEC_SIZE)
    # B = generate_rand_vec(VEC_SIZE)
    #
    # acc_loop = []
    # acc_threads = []
    # for _ in range(10):
    #     _, elapsed = executors.cpu_simple_loop(A, B)
    #     acc_loop.append(elapsed)
    #     _, elapsed = executors.cpu_with_threads(A, B)
    #     acc_threads.append(elapsed)
    # print(f"Mean time with simple loop: {sum(acc_loop) / len(acc_loop):.2f}s")
    # print(f"Mean time with threads: {sum(acc_threads) / len(acc_threads):.2f}s")

    # executors.gpu_triton_baseline(VEC_SIZE)
    # executors.gpu_triton_with_copy(VEC_SIZE)

    executors.cpu_pytorch(VEC_SIZE)
    # executors.gpu_pytorch(VEC_SIZE)

    print("END")


if __name__ == '__main__':
    main()
