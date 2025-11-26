from src import executors


def main() -> None:
    # executors.simple_loop()
    executors.pytorch_cpu()
    # executors.pytorch_gpu()
    executors.triton_baseline()
    executors.faster_triton()

    print("END")


if __name__ == '__main__':
    main()
