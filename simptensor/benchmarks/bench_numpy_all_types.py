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

def bench_matmul(N, dtype_name, dtype, iterations, is_float=False):
    """Benchmark N×N matmul for given dtype"""
    # Initialize matrices
    if is_float:
        A = np.random.randn(N, N).astype(dtype)
        B = np.random.randn(N, N).astype(dtype)
    else:
        A = np.arange(N * N, dtype=np.int64).reshape(N, N) % 256
        B = np.arange(N * N, dtype=np.int64).reshape(N, N).T % 256
        A = A.astype(dtype)
        B = B.astype(dtype)
    
    def matmul_func():
        C = np.matmul(A, B)
        return C.sum()
    
    time_ms = benchmark(matmul_func, iterations)
    result = matmul_func()
    
    if is_float:
        giops = 2.0 * N * N * N / 1e9
        giops_per_sec = giops / (time_ms / 1000.0)
        print(f"{dtype_name:4s} {N:3d}×{N:4d} │ {time_ms:8.3f} ms │ {giops_per_sec:8.2f} GFLOP/s │  ✓")
    else:
        giops = 2.0 * N * N * N / 1e9
        giops_per_sec = giops / (time_ms / 1000.0)
        print(f"{dtype_name:4s} {N:3d}×{N:4d} │ {time_ms:8.3f} ms │ {giops_per_sec:8.2f} GIOP/s  │  ✓")

def main():
    print("═" * 79)
    print("   NumPy Matrix Multiplication Benchmarks (All Types)")
    print("═" * 79)
    print()
    
    # Float types
    print("Float Types:")
    print(" Type Size   │   Time    │  Performance │ Status")
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 14 + "┼" + "─" * 8)
    
    bench_matmul(64, "f32", np.float32, 10, is_float=True)
    bench_matmul(128, "f32", np.float32, 10, is_float=True)
    bench_matmul(256, "f32", np.float32, 5, is_float=True)
    bench_matmul(512, "f32", np.float32, 3, is_float=True)
    bench_matmul(1024, "f32", np.float32, 2, is_float=True)
    
    print()
    print("Integer Types:")
    print(" Type Size   │   Time    │  Performance │ Status")
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 14 + "┼" + "─" * 8)
    
    # i8
    bench_matmul(64, "i8", np.int8, 10, is_float=False)
    bench_matmul(128, "i8", np.int8, 10, is_float=False)
    bench_matmul(256, "i8", np.int8, 5, is_float=False)
    bench_matmul(512, "i8", np.int8, 3, is_float=False)
    
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 14 + "┼" + "─" * 8)
    
    # i16
    bench_matmul(64, "i16", np.int16, 10, is_float=False)
    bench_matmul(128, "i16", np.int16, 10, is_float=False)
    bench_matmul(256, "i16", np.int16, 5, is_float=False)
    bench_matmul(512, "i16", np.int16, 3, is_float=False)
    
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 14 + "┼" + "─" * 8)
    
    # i32
    bench_matmul(64, "i32", np.int32, 10, is_float=False)
    bench_matmul(128, "i32", np.int32, 10, is_float=False)
    bench_matmul(256, "i32", np.int32, 5, is_float=False)
    bench_matmul(512, "i32", np.int32, 3, is_float=False)
    bench_matmul(1024, "i32", np.int32, 2, is_float=False)
    
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 14 + "┼" + "─" * 8)
    
    # i64
    bench_matmul(256, "i64", np.int64, 5, is_float=False)
    
    print("═" * 79)
    print()
    print("Notes:")
    print("• NumPy uses optimized BLAS backend (OpenBLAS/MKL)")
    print("• Integer types may overflow without proper accumulator width")
    print()

if __name__ == "__main__":
    main()
