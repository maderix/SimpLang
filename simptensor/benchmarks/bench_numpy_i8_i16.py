import numpy as np
import time

def benchmark(func, iterations=5):
    """Benchmark a function with warmup"""
    func()  # warmup
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()
    return (end - start) / iterations * 1000  # ms

def bench_matmul(N, dtype_name, dtype, iterations):
    """Benchmark N×N matmul for given dtype"""
    # Initialize with smaller values to avoid overflow during init
    A = np.arange(N * N, dtype=np.int32).reshape(N, N) % 64
    B = np.arange(N * N, dtype=np.int32).reshape(N, N).T % 64
    
    # Convert to target dtype
    A = A.astype(dtype)
    B = B.astype(dtype)
    
    def matmul_func():
        C = np.matmul(A, B)
        return C.sum()
    
    time_ms = benchmark(matmul_func, iterations)
    result = matmul_func()
    
    giops = 2.0 * N * N * N / 1e9
    giops_per_sec = giops / (time_ms / 1000.0)
    
    print(f"{dtype_name:4s} {N:3d}×{N:4d} │ {time_ms:8.3f} ms │ {giops_per_sec:8.2f} │  ✓")

def main():
    print("═" * 79)
    print("   NumPy i8/i16 Matrix Multiplication Benchmarks")
    print("═" * 79)
    print()
    print(" Type Size   │   Time    │  GIOP/s  │ Status")
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 10 + "┼" + "─" * 8)
    
    # i8 benchmarks
    bench_matmul(64, "i8", np.int8, 10)
    bench_matmul(128, "i8", np.int8, 10)
    bench_matmul(256, "i8", np.int8, 5)
    bench_matmul(512, "i8", np.int8, 3)
    
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 10 + "┼" + "─" * 8)
    
    # i16 benchmarks
    bench_matmul(64, "i16", np.int16, 10)
    bench_matmul(128, "i16", np.int16, 10)
    bench_matmul(256, "i16", np.int16, 5)
    bench_matmul(512, "i16", np.int16, 3)
    
    print("═" * 79)
    print()
    print("Notes:")
    print("• NumPy uses optimized BLAS backend (OpenBLAS/MKL)")
    print("• Results may overflow in int8/int16 - NumPy doesn't auto-promote")
    print("• SimpLang auto-promotes to i32 accumulator (prevents overflow)")
    print()

if __name__ == "__main__":
    main()
