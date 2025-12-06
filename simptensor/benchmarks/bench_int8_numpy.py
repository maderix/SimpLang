#!/usr/bin/env python3
"""
INT8 MatMul Benchmark - NumPy Reference

Benchmarks numpy INT8 matmul for comparison with SimpLang/C++ implementations.
Outputs CSV compatible with roofline plotting.

Usage:
    python3 bench_int8_numpy.py
"""

import numpy as np
import time
import csv

def benchmark_int8_matmul(N, iterations=10, warmup=3):
    """Benchmark INT8 matmul for NxN matrices."""
    # Initialize matrices (same values as C++ reference)
    A = np.zeros((N, N), dtype=np.int8)
    B = np.zeros((N, N), dtype=np.int8)

    for i in range(N):
        for j in range(N):
            val = ((i * N + j) % 127) - 64
            A[i, j] = val
            B[j, i] = val  # Transposed initialization

    # Warmup
    for _ in range(warmup):
        # NumPy doesn't have native INT8 matmul, cast to INT32
        C = np.matmul(A.astype(np.int32), B.astype(np.int32))

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = np.matmul(A.astype(np.int32), B.astype(np.int32))
        end = time.perf_counter()
        times.append(end - start)

    avg_time_ms = np.mean(times) * 1000
    checksum = int(C.sum())

    # Compute GIOP/s
    ops = 2.0 * N * N * N
    giops = ops / 1e9 / (avg_time_ms / 1000)

    # Arithmetic intensity for roofline
    memory_bytes = 2 * N * N * 1 + N * N * 4  # A,B are INT8, C is INT32
    arithmetic_intensity = ops / memory_bytes

    return {
        'N': N,
        'time_ms': avg_time_ms,
        'giops': giops,
        'checksum': checksum,
        'arithmetic_intensity': arithmetic_intensity,
        'memory_bytes': memory_bytes
    }

def main():
    print("=" * 70)
    print("   INT8 MatMul Benchmark - NumPy Reference")
    print("=" * 70)
    print(f"   NumPy version: {np.__version__}")
    print()

    sizes = [32, 64, 128, 256, 384, 512, 768, 1024, 2048, 4096]
    iterations_map = {32: 100, 64: 50, 128: 20, 256: 10, 384: 5, 512: 3, 768: 2, 1024: 1, 2048: 1, 4096: 1}

    results = []

    print(f"{'Size':>10} | {'Time (ms)':>12} | {'GIOP/s':>10} | {'Checksum':>15}")
    print("-" * 55)

    for N in sizes:
        iters = iterations_map.get(N, 1)
        result = benchmark_int8_matmul(N, iterations=max(1, iters), warmup=2)
        results.append(result)
        print(f"{N:>5}x{N:<4} | {result['time_ms']:>12.3f} | {result['giops']:>10.2f} | {result['checksum']:>15}")

    print("=" * 70)

    # Write CSV
    csv_file = '/tmp/int8_numpy_benchmark.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['N', 'time_ms', 'giops', 'checksum', 'arithmetic_intensity', 'memory_bytes'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV written to: {csv_file}")

if __name__ == '__main__':
    main()
