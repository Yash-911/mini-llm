"""CPU vs GPU (MPS/CUDA) matmul benchmark across matrix sizes.

Usage:  uv run python scripts/bench_matmul.py
"""
import time

import torch

from mini_llm.utils import get_device, sync

SIZES = [32, 128, 512, 1024, 2048, 4096]
WARMUP, ITERS = 3, 10


def bench(device: str, n: int) -> float:
    """Average milliseconds for one (n, n) @ (n, n) matmul on `device`."""
    a = torch.randn(n, n, device=device)  # (n, n)
    b = torch.randn(n, n, device=device)  # (n, n)
    for _ in range(WARMUP):  # first calls pay one-off costs (kernel compile, allocation)
        a @ b
    sync(device)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        a @ b
    sync(device)  # without this, GPU timing only measures how fast we *queued* the work
    return (time.perf_counter() - t0) / ITERS * 1000


def main():
    gpu = get_device()
    print(f"GPU device: {gpu}\n")
    print(f"{'n':>6} | {'cpu ms':>9} | {gpu + ' ms':>9} | {'speedup':>8} | {gpu + ' GFLOP/s':>14}")
    print("-" * 60)
    for n in SIZES:
        cpu_ms = bench("cpu", n)
        gpu_ms = bench(gpu, n) if gpu != "cpu" else cpu_ms
        flops = 2 * n**3  # each of n*n outputs = n multiplies + n adds
        gflops = flops / (gpu_ms / 1000) / 1e9
        print(f"{n:>6} | {cpu_ms:>9.3f} | {gpu_ms:>9.3f} | {cpu_ms / gpu_ms:>7.1f}x | {gflops:>14.0f}")


if __name__ == "__main__":
    main()
