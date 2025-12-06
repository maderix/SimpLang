#!/usr/bin/env python3
"""
INT8 MatMul Benchmark - TensorFlow / XLA Reference

Benchmarks TensorFlow INT8 matmul for comparison with SimpLang/C++ implementations.
Tests both multi-threaded (oneDNN) and single-threaded modes.

Usage:
    pip install tensorflow
    python3 bench_int8_tflite.py
"""

import numpy as np
import time
import csv
import os

# Set single-threaded mode BEFORE importing TensorFlow
SINGLE_THREADED = os.environ.get('TF_SINGLE_THREAD', '0') == '1'
if SINGLE_THREADED:
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

try:
    import tensorflow as tf
    if SINGLE_THREADED:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    print(f"TensorFlow version: {tf.__version__}")
    HAS_TF = True
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    HAS_TF = False

def benchmark_tf_int8_matmul(N, iterations=10, warmup=3):
    """Benchmark TensorFlow INT8 matmul for NxN matrices."""
    # Initialize matrices (same values as C++ reference)
    A_np = np.zeros((N, N), dtype=np.int8)
    B_np = np.zeros((N, N), dtype=np.int8)

    for i in range(N):
        for j in range(N):
            val = ((i * N + j) % 127) - 64
            A_np[i, j] = val
            B_np[j, i] = val  # Transposed initialization

    # Convert to TF tensors - TF doesn't support int8 matmul directly,
    # so we use int32 (similar to what XLA would do internally)
    A = tf.constant(A_np, dtype=tf.int32)
    B = tf.constant(B_np, dtype=tf.int32)

    # Warmup
    for _ in range(warmup):
        C = tf.linalg.matmul(A, B)
        _ = C.numpy()  # Force execution

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = tf.linalg.matmul(A, B)
        _ = C.numpy()  # Force execution
        end = time.perf_counter()
        times.append(end - start)

    avg_time_ms = np.mean(times) * 1000
    checksum = int(tf.reduce_sum(C).numpy())

    # Compute GIOP/s
    ops = 2.0 * N * N * N
    giops = ops / 1e9 / (avg_time_ms / 1000)

    # Arithmetic intensity for roofline
    memory_bytes = 2 * N * N * 1 + N * N * 4  # A,B are INT8 (input), C is INT32
    arithmetic_intensity = ops / memory_bytes

    return {
        'N': N,
        'time_ms': avg_time_ms,
        'giops': giops,
        'checksum': checksum,
        'arithmetic_intensity': arithmetic_intensity,
        'memory_bytes': memory_bytes
    }

@tf.function(jit_compile=True)
def xla_matmul(A, B):
    """XLA-compiled matmul."""
    return tf.linalg.matmul(A, B)

def benchmark_xla_int8_matmul(N, iterations=10, warmup=3):
    """Benchmark XLA-compiled INT8 matmul for NxN matrices."""
    A_np = np.zeros((N, N), dtype=np.int8)
    B_np = np.zeros((N, N), dtype=np.int8)

    for i in range(N):
        for j in range(N):
            val = ((i * N + j) % 127) - 64
            A_np[i, j] = val
            B_np[j, i] = val

    A = tf.constant(A_np, dtype=tf.int32)
    B = tf.constant(B_np, dtype=tf.int32)

    # Warmup (includes JIT compilation)
    for _ in range(warmup + 2):
        C = xla_matmul(A, B)
        _ = C.numpy()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = xla_matmul(A, B)
        _ = C.numpy()
        end = time.perf_counter()
        times.append(end - start)

    avg_time_ms = np.mean(times) * 1000
    checksum = int(tf.reduce_sum(C).numpy())

    ops = 2.0 * N * N * N
    giops = ops / 1e9 / (avg_time_ms / 1000)
    memory_bytes = 2 * N * N * 1 + N * N * 4
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
    if not HAS_TF:
        return

    print("=" * 80)
    print("   INT8 MatMul Benchmark - TensorFlow / XLA Reference")
    print("=" * 80)

    # Disable GPU to ensure CPU execution
    tf.config.set_visible_devices([], 'GPU')

    threads = tf.config.threading.get_intra_op_parallelism_threads()
    thread_str = "single-threaded" if threads == 1 else f"multi-threaded ({os.cpu_count()} cores)"
    print(f"Running on CPU only ({thread_str})")
    print()

    sizes = [32, 64, 128, 256, 384, 512, 768, 1024, 2048, 4096]
    iterations_map = {32: 100, 64: 50, 128: 20, 256: 10, 384: 5, 512: 3, 768: 2, 1024: 2, 2048: 1, 4096: 1}

    tf_results = []
    xla_results = []

    print("=== TensorFlow Eager Mode ===")
    print(f"{'Size':>10} | {'Time (ms)':>12} | {'GIOP/s':>10} | {'Checksum':>15}")
    print("-" * 55)

    for N in sizes:
        iters = iterations_map.get(N, 1)
        try:
            result = benchmark_tf_int8_matmul(N, iterations=max(1, iters), warmup=2)
            tf_results.append(result)
            print(f"{N:>5}x{N:<4} | {result['time_ms']:>12.3f} | {result['giops']:>10.2f} | {result['checksum']:>15}")
        except Exception as e:
            print(f"{N:>5}x{N:<4} | ERROR: {e}")

    print()
    print("=== TensorFlow XLA (JIT Compiled) ===")
    print(f"{'Size':>10} | {'Time (ms)':>12} | {'GIOP/s':>10} | {'Checksum':>15}")
    print("-" * 55)

    for N in sizes:
        iters = iterations_map.get(N, 1)
        try:
            result = benchmark_xla_int8_matmul(N, iterations=max(1, iters), warmup=2)
            xla_results.append(result)
            print(f"{N:>5}x{N:<4} | {result['time_ms']:>12.3f} | {result['giops']:>10.2f} | {result['checksum']:>15}")
        except Exception as e:
            print(f"{N:>5}x{N:<4} | ERROR: {e}")

    print("=" * 80)

    # Write CSV
    csv_file = '/tmp/int8_tflite_benchmark.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'tf_time_ms', 'tf_giops', 'xla_time_ms', 'xla_giops', 'arithmetic_intensity'])
        for tf_r, xla_r in zip(tf_results, xla_results):
            writer.writerow([
                tf_r['N'],
                tf_r['time_ms'], tf_r['giops'],
                xla_r['time_ms'], xla_r['giops'],
                tf_r['arithmetic_intensity']
            ])

    print(f"\nCSV written to: {csv_file}")

if __name__ == '__main__':
    main()
