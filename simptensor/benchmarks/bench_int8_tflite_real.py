#!/usr/bin/env python3
"""
INT8 MatMul Benchmark - TFLite Real INT8 Quantized

Uses TFLite with actual INT8 quantized operations via XNNPACK.

Usage:
    pip install tensorflow
    python3 bench_int8_tflite_real.py
"""

import numpy as np
import time
import csv
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

def create_int8_matmul_tflite_model(M, K, N):
    """Create a TFLite model with INT8 quantized matmul."""

    # Create a simple model: matmul(A, B)
    class MatMulModel(tf.Module):
        def __init__(self):
            super().__init__()

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[M, K], dtype=tf.float32),
            tf.TensorSpec(shape=[K, N], dtype=tf.float32)
        ])
        def matmul(self, a, b):
            return tf.matmul(a, b)

    model = MatMulModel()

    # Convert to TFLite with INT8 quantization
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [model.matmul.get_concrete_function()],
        model
    )

    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Representative dataset for quantization calibration
    def representative_dataset():
        for _ in range(100):
            a = np.random.uniform(-1, 1, (M, K)).astype(np.float32)
            b = np.random.uniform(-1, 1, (K, N)).astype(np.float32)
            yield [a, b]

    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    return tflite_model

def benchmark_tflite_int8(N, iterations=10, warmup=3):
    """Benchmark TFLite INT8 matmul."""

    print(f"  Creating INT8 TFLite model for {N}x{N}...", end=" ", flush=True)

    try:
        tflite_model = create_int8_matmul_tflite_model(N, N, N)
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    # Create interpreter with XNNPACK delegate for optimized INT8
    try:
        interpreter = tf.lite.Interpreter(
            model_content=tflite_model,
            num_threads=1  # Single-threaded for fair comparison
        )
    except Exception as e:
        print(f"Interpreter FAILED: {e}")
        return None

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"OK (inputs: {[d['dtype'] for d in input_details]})")

    # Prepare INT8 inputs
    # Get quantization parameters
    a_scale = input_details[0]['quantization'][0]
    a_zero = input_details[0]['quantization'][1]
    b_scale = input_details[1]['quantization'][0]
    b_zero = input_details[1]['quantization'][1]

    # Create test data (matching our C++ benchmark pattern)
    A_fp = np.zeros((N, N), dtype=np.float32)
    B_fp = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            val = (((i * N + j) % 127) - 64) / 127.0  # Normalized to [-0.5, 0.5]
            A_fp[i, j] = val
            B_fp[j, i] = val

    # Quantize to INT8
    A_i8 = np.clip(np.round(A_fp / a_scale + a_zero), -128, 127).astype(np.int8)
    B_i8 = np.clip(np.round(B_fp / b_scale + b_zero), -128, 127).astype(np.int8)

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]['index'], A_i8)
        interpreter.set_tensor(input_details[1]['index'], B_i8)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], A_i8)
        interpreter.set_tensor(input_details[1]['index'], B_i8)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        end = time.perf_counter()
        times.append(end - start)

    avg_time_ms = np.mean(times) * 1000

    # Compute GIOP/s (2*N^3 operations)
    ops = 2.0 * N * N * N
    giops = ops / 1e9 / (avg_time_ms / 1000)

    return {
        'N': N,
        'time_ms': avg_time_ms,
        'giops': giops,
        'input_dtype': str(input_details[0]['dtype']),
        'output_dtype': str(output_details[0]['dtype'])
    }

def main():
    print("=" * 80)
    print("   INT8 MatMul Benchmark - TFLite Real INT8 Quantized")
    print("=" * 80)
    print()

    # Disable GPU
    tf.config.set_visible_devices([], 'GPU')

    sizes = [32, 64, 128, 256, 512, 1024]
    iterations_map = {32: 100, 64: 50, 128: 20, 256: 10, 512: 5, 1024: 3}

    results = []

    print(f"{'Size':>10} | {'Time (ms)':>12} | {'GIOP/s':>10} | {'Input':>10} | {'Output':>10}")
    print("-" * 65)

    for N in sizes:
        iters = iterations_map.get(N, 3)
        result = benchmark_tflite_int8(N, iterations=iters, warmup=2)

        if result:
            results.append(result)
            print(f"{N:>5}x{N:<4} | {result['time_ms']:>12.3f} | {result['giops']:>10.2f} | {result['input_dtype']:>10} | {result['output_dtype']:>10}")
        else:
            print(f"{N:>5}x{N:<4} | {'FAILED':>12} |")

    print("=" * 80)

    if results:
        csv_file = '/tmp/int8_tflite_real_benchmark.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['N', 'time_ms', 'giops', 'input_dtype', 'output_dtype'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV written to: {csv_file}")

if __name__ == '__main__':
    main()
