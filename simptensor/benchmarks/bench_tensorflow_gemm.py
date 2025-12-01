#!/usr/bin/env python3
"""
TensorFlow GEMM Benchmark for ARM (without TFLite conversion)
Direct matmul benchmark using TensorFlow ops
"""
import numpy as np
import time
import tensorflow as tf

# Disable GPU (if any)
tf.config.set_visible_devices([], 'GPU')

def benchmark(func, iterations=5):
    """Benchmark a function with warmup"""
    func()  # warmup
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()
    return (end - start) / iterations * 1000  # ms

def bench_tf_gemm(N, dtype_name, dtype_tf, dtype_np, iterations, is_float=True):
    """Benchmark TensorFlow GEMM for given size and dtype"""

    # Create test data
    if is_float:
        A = tf.constant(np.random.randn(N, N).astype(dtype_np), dtype=dtype_tf)
        B = tf.constant(np.random.randn(N, N).astype(dtype_np), dtype=dtype_tf)
    else:
        A_np = np.arange(N * N, dtype=np.int64).reshape(N, N) % 256
        B_np = np.arange(N * N, dtype=np.int64).reshape(N, N).T % 256
        A = tf.constant(A_np.astype(dtype_np), dtype=dtype_tf)
        B = tf.constant(B_np.astype(dtype_np), dtype=dtype_tf)

    @tf.function
    def run_matmul():
        C = tf.matmul(A, B)
        return tf.reduce_sum(C)

    # Compile the function first
    _ = run_matmul()

    def run_inference():
        result = run_matmul()
        return result.numpy()

    time_ms = benchmark(run_inference, iterations)
    result = run_inference()

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
    print("   TensorFlow GEMM Benchmarks (ARM CPU)")
    print("═" * 79)
    print()

    # Float types
    print("Float Types (f32):")
    print(" Type Size   │   Time    │  Performance │ Status")
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 14 + "┼" + "─" * 8)

    bench_tf_gemm(64, "f32", tf.float32, np.float32, 10, is_float=True)
    bench_tf_gemm(128, "f32", tf.float32, np.float32, 10, is_float=True)
    bench_tf_gemm(256, "f32", tf.float32, np.float32, 5, is_float=True)
    bench_tf_gemm(512, "f32", tf.float32, np.float32, 3, is_float=True)
    bench_tf_gemm(1024, "f32", tf.float32, np.float32, 2, is_float=True)

    print()
    print("Integer Types (i32):")
    print(" Type Size   │   Time    │  Performance │ Status")
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 14 + "┼" + "─" * 8)

    bench_tf_gemm(64, "i32", tf.int32, np.int32, 10, is_float=False)
    bench_tf_gemm(128, "i32", tf.int32, np.int32, 10, is_float=False)
    bench_tf_gemm(256, "i32", tf.int32, np.int32, 5, is_float=False)
    bench_tf_gemm(512, "i32", tf.int32, np.int32, 3, is_float=False)
    bench_tf_gemm(1024, "i32", tf.int32, np.int32, 2, is_float=False)

    print("═" * 79)
    print()
    print("Notes:")
    print("• TensorFlow CPU-only (no GPU)")
    print("• Uses Eigen backend for matrix operations")
    print("• @tf.function compilation enabled")
    print()

if __name__ == "__main__":
    main()
