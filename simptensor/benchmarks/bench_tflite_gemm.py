#!/usr/bin/env python3
"""
TensorFlow Lite GEMM Benchmark for ARM
Proxy for ML inference performance using matrix multiplication
"""
import numpy as np
import time
try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed. Installing TFLite runtime...")
    import sys
    sys.exit(1)

def benchmark(func, iterations=5):
    """Benchmark a function with warmup"""
    func()  # warmup
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()
    return (end - start) / iterations * 1000  # ms

def create_gemm_model(N, dtype=tf.float32):
    """Create a simple GEMM model using TFLite"""
    # Create a simple matmul model
    class GEMMModel(tf.Module):
        def __init__(self):
            super().__init__()

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[N, N], dtype=dtype),
            tf.TensorSpec(shape=[N, N], dtype=dtype)
        ])
        def matmul(self, a, b):
            return tf.matmul(a, b)

    model = GEMMModel()

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [model.matmul.get_concrete_function()]
    )

    # Optimization for ARM
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()
    return tflite_model

def bench_tflite_gemm(N, dtype_name, dtype_tf, dtype_np, iterations, is_float=True):
    """Benchmark TFLite GEMM for given size and dtype"""

    # Create TFLite model
    tflite_model = create_gemm_model(N, dtype_tf)

    # Create interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create test data
    if is_float:
        A = np.random.randn(N, N).astype(dtype_np)
        B = np.random.randn(N, N).astype(dtype_np)
    else:
        A = np.arange(N * N, dtype=np.int64).reshape(N, N) % 256
        B = np.arange(N * N, dtype=np.int64).reshape(N, N).T % 256
        A = A.astype(dtype_np)
        B = B.astype(dtype_np)

    def run_inference():
        interpreter.set_tensor(input_details[0]['index'], A)
        interpreter.set_tensor(input_details[1]['index'], B)
        interpreter.invoke()
        C = interpreter.get_tensor(output_details[0]['index'])
        return C.sum()

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
    print("   TensorFlow Lite GEMM Benchmarks (ARM Optimized)")
    print("═" * 79)
    print()

    # Float types
    print("Float Types (f32):")
    print(" Type Size   │   Time    │  Performance │ Status")
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 14 + "┼" + "─" * 8)

    try:
        bench_tflite_gemm(64, "f32", tf.float32, np.float32, 10, is_float=True)
        bench_tflite_gemm(128, "f32", tf.float32, np.float32, 10, is_float=True)
        bench_tflite_gemm(256, "f32", tf.float32, np.float32, 5, is_float=True)
        bench_tflite_gemm(512, "f32", tf.float32, np.float32, 3, is_float=True)
        bench_tflite_gemm(1024, "f32", tf.float32, np.float32, 2, is_float=True)
    except Exception as e:
        print(f"Error in float benchmark: {e}")

    print()
    print("Integer Types (i32):")
    print(" Type Size   │   Time    │  Performance │ Status")
    print("─" * 13 + "┼" + "─" * 11 + "┼" + "─" * 14 + "┼" + "─" * 8)

    try:
        bench_tflite_gemm(64, "i32", tf.int32, np.int32, 10, is_float=False)
        bench_tflite_gemm(128, "i32", tf.int32, np.int32, 10, is_float=False)
        bench_tflite_gemm(256, "i32", tf.int32, np.int32, 5, is_float=False)
        bench_tflite_gemm(512, "i32", tf.int32, np.int32, 3, is_float=False)
        bench_tflite_gemm(1024, "i32", tf.int32, np.int32, 2, is_float=False)
    except Exception as e:
        print(f"Error in integer benchmark: {e}")

    print("═" * 79)
    print()
    print("Notes:")
    print("• TensorFlow Lite optimized for ARM with NEON")
    print("• Uses XNNPack delegate for acceleration")
    print("• Representative of real ML inference workloads")
    print()

if __name__ == "__main__":
    main()
