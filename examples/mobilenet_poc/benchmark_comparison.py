#!/usr/bin/env python3

import onnxruntime as ort
import numpy as np
import time
import sys
import os
import subprocess
import json

def benchmark_onnxruntime(model_path, num_runs=100, warmup_runs=10):
    print(f"Loading ONNX model: {model_path}")
    
    # Create ONNX Runtime session with optimizations
    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_shape = [dim if isinstance(dim, int) else 1 for dim in input_shape]
    
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    
    # Create random input
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup runs
    print(f"\nWarming up with {warmup_runs} runs...")
    for _ in range(warmup_runs):
        outputs = session.run(None, {input_name: input_data})
    
    # Benchmark runs
    print(f"\nBenchmarking with {num_runs} runs...")
    times = []
    results = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        outputs = session.run(None, {input_name: input_data})
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)
        
        # Get argmax for classification
        output = outputs[0].flatten()
        predicted_class = np.argmax(output)
        max_score = output[predicted_class]
        results.append((predicted_class, max_score))
        
        if i % 20 == 0:
            print(f"  Run {i:3d}: {elapsed_ms:.3f} ms, Class: {predicted_class}, Score: {max_score:.4f}")
    
    # Calculate statistics
    times_array = np.array(times)
    stats = {
        'mean_time': np.mean(times_array),
        'std_time': np.std(times_array),
        'median_time': np.median(times_array),
        'min_time': np.min(times_array),
        'max_time': np.max(times_array),
        'p95_time': np.percentile(times_array, 95),
        'p99_time': np.percentile(times_array, 99),
        'throughput': 1000/np.mean(times_array),
        'results': results
    }
    
    return stats

def benchmark_simplang(num_runs=100, warmup_runs=10):
    print("\n" + "="*60)
    print("Benchmarking SimpleLang+SimpBLAS Implementation")
    print("="*60)
    
    # First compile the SimpleLang code if needed
    sl_file = "mobilenetv2_full.sl"
    so_file = "mobilenetv2_full.so"
    
    if not os.path.exists(so_file) or os.path.getmtime(sl_file) > os.path.getmtime(so_file):
        print(f"Compiling {sl_file}...")
        compile_cmd = f"../build/src/simplang {sl_file} -o {so_file}"
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return None
        print("Compilation successful!")
    
    # Build the runner if needed
    runner_exe = "./mobilenet_runner"
    if not os.path.exists(runner_exe):
        print("Building runner executable...")
        build_cmd = "g++ -o mobilenet_runner mobilenet_host_loader.cpp -ldl -lsimpblas -L/usr/local/lib -Wl,-rpath,/usr/local/lib"
        result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return None
    
    # Set environment for running with SimpBLAS
    env = os.environ.copy()
    env['LD_PRELOAD'] = '../simpblas/build/libsimpblas.so'
    
    print(f"\nWarming up with {warmup_runs} runs...")
    for _ in range(warmup_runs):
        result = subprocess.run([runner_exe], capture_output=True, text=True, env=env)
    
    print(f"\nBenchmarking with {num_runs} runs...")
    times = []
    results = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        result = subprocess.run([runner_exe], capture_output=True, text=True, env=env)
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)
        
        # Parse output to get result and actual inference time
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            if "Classification result:" in line:
                # Extract the result value
                result_val = float(line.split(':')[1].strip())
                results.append(result_val)
            elif "Inference time:" in line and "ms" in line:
                # Extract the actual kernel execution time (not including dlopen)
                internal_time = float(line.split(':')[1].replace('ms', '').strip())
                # Use internal kernel time
                times[-1] = internal_time
        
        if i % 20 == 0:
            print(f"  Run {i:3d}: {times[-1]:.3f} ms")
    
    # Calculate statistics
    times_array = np.array(times)
    stats = {
        'mean_time': np.mean(times_array),
        'std_time': np.std(times_array),
        'median_time': np.median(times_array),
        'min_time': np.min(times_array),
        'max_time': np.max(times_array),
        'p95_time': np.percentile(times_array, 95),
        'p99_time': np.percentile(times_array, 99),
        'throughput': 1000/np.mean(times_array),
        'results': results
    }
    
    return stats

def print_stats(name, stats):
    print(f"\n{name} Performance Summary:")
    print("="*60)
    print(f"Runs: {len(stats.get('results', []))}")
    print(f"Mean time: {stats['mean_time']:.3f} ms ± {stats['std_time']:.3f} ms")
    print(f"Median time: {stats['median_time']:.3f} ms")
    print(f"Min time: {stats['min_time']:.3f} ms")
    print(f"Max time: {stats['max_time']:.3f} ms")
    print(f"P95 time: {stats['p95_time']:.3f} ms")
    print(f"P99 time: {stats['p99_time']:.3f} ms")
    print(f"Throughput: {stats['throughput']:.2f} FPS")

def compare_accuracy():
    """Compare accuracy between ONNX Runtime and SimpleLang"""
    print("\n" + "="*60)
    print("Accuracy Comparison")
    print("="*60)
    
    # Create same input for both
    np.random.seed(42)  # Fixed seed for reproducibility
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Save input for SimpleLang to use
    print("Saving input data for SimpleLang...")
    input_data.tofile('test_input.bin')
    
    # Run ONNX Runtime
    print("\nRunning ONNX Runtime...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession("mobilenetv2-7.onnx", sess_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    onnx_output = outputs[0].flatten()
    
    # Get ONNX predictions
    onnx_top_class = np.argmax(onnx_output)
    onnx_top_score = onnx_output[onnx_top_class]
    top5_indices = np.argsort(onnx_output)[-5:][::-1]
    top5_scores = onnx_output[top5_indices]
    
    print(f"ONNX Runtime Results:")
    print(f"  Top class: {onnx_top_class} (score: {onnx_top_score:.4f})")
    print(f"  Top-5 classes: {top5_indices.tolist()}")
    print(f"  Output range: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
    print(f"  Output mean: {onnx_output.mean():.4f}, std: {onnx_output.std():.4f}")
    
    # Run SimpleLang - currently returns single value
    print("\nRunning SimpleLang+SimpBLAS...")
    env = os.environ.copy()
    env['LD_PRELOAD'] = '../simpblas/build/libsimpblas.so'
    result = subprocess.run(['./mobilenet_runner'], capture_output=True, text=True, env=env)
    
    simplang_result = None
    for line in result.stdout.strip().split('\n'):
        if "Classification result:" in line:
            simplang_result = float(line.split(':')[1].strip())
            break
    
    print(f"SimpleLang Results:")
    print(f"  Result: {simplang_result}")
    
    print("\n" + "="*60)
    print("Accuracy Analysis:")
    print("="*60)
    
    # Save ONNX logits for detailed comparison
    print("Saving ONNX logits for analysis...")
    np.save('onnx_logits.npy', onnx_output)
    
    if simplang_result is not None:
        # Decode SimpleLang result: class_id + mean*0.01 + std*0.0001
        simplang_class = int(simplang_result)
        encoded_mean = (simplang_result - simplang_class) / 0.01
        encoded_std = ((simplang_result - simplang_class) % 0.01) / 0.0001
        
        print(f"\nDetailed Comparison:")
        print(f"  ONNX predicted class: {onnx_top_class}")
        print(f"  SimpleLang predicted class: {simplang_class}")
        print(f"  SimpleLang logits mean: {encoded_mean:.4f}")
        print(f"  SimpleLang logits variance: {encoded_std:.4f}")
        
        if simplang_class == onnx_top_class:
            print(f"  ✅ EXACT MATCH! Both models predict class {simplang_class}")
        elif simplang_class in top5_indices:
            rank = np.where(top5_indices == simplang_class)[0][0] + 1
            score = onnx_output[simplang_class]
            print(f"  🟡 SimpleLang class {simplang_class} is #{rank} in ONNX top-5 (score: {score:.4f})")
        else:
            score = onnx_output[simplang_class] if simplang_class < 1000 else "N/A"
            print(f"  ❌ SimpleLang class {simplang_class} not in ONNX top-5 (ONNX score: {score})")
        
        # Compute logits comparison metrics
        print(f"\nLogits Statistics Comparison:")
        onnx_mean = onnx_output.mean()
        onnx_std = onnx_output.std()
        onnx_var = onnx_output.var()
        
        print(f"  ONNX logits:")
        print(f"    Range: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
        print(f"    Mean: {onnx_mean:.4f}")
        print(f"    Std: {onnx_std:.4f}")
        print(f"    Variance: {onnx_var:.4f}")
        print(f"    Top-5 scores: {top5_scores}")
        
        print(f"  SimpleLang logits:")
        print(f"    Mean: {encoded_mean:.4f}")
        print(f"    Variance: {encoded_std:.4f}")
        print(f"    Predicted class score in ONNX: {onnx_output[simplang_class]:.4f}")
        
        # Compare statistics
        mean_diff = abs(encoded_mean - onnx_mean)
        var_diff = abs(encoded_std - onnx_var)
        
        print(f"\nStatistical Differences:")
        print(f"  Mean difference: {mean_diff:.4f}")
        print(f"  Variance difference: {var_diff:.4f}")
        
        if mean_diff < 0.1 and var_diff < 1.0:
            print(f"  📊 Good statistical similarity - logits distributions are close")
        elif mean_diff < 0.5:
            print(f"  📊 Moderate statistical similarity - some differences in logits")  
        else:
            print(f"  📊 Poor statistical similarity - significant logits differences")
            
        print(f"\nNOTE: For exact cosine similarity and MSE, need full SimpleLang logits vector")
        print(f"      Current approach only compares basic statistics")
    
    # Compare with different input patterns
    print("\nTesting with different input patterns:")
    test_patterns = {
        'zeros': np.zeros((1, 3, 224, 224), dtype=np.float32),
        'ones': np.ones((1, 3, 224, 224), dtype=np.float32),
        'random': input_data
    }
    
    for pattern_name, test_input in test_patterns.items():
        outputs = session.run(None, {input_name: test_input})
        onnx_out = outputs[0].flatten()
        print(f"  {pattern_name}: ONNX argmax={np.argmax(onnx_out)}, max={onnx_out.max():.4f}")

def compare_implementations():
    print("\n" + "="*60)
    print("MobileNetV2 Performance Comparison")
    print("="*60)
    
    # Benchmark ONNX Runtime
    print("\n[1/2] Running ONNX Runtime benchmark...")
    onnx_stats = benchmark_onnxruntime("mobilenetv2-7.onnx", num_runs=100)
    print_stats("ONNX Runtime", onnx_stats)
    
    # Benchmark SimpleLang
    print("\n[2/2] Running SimpleLang+SimpBLAS benchmark...")
    simplang_stats = benchmark_simplang(num_runs=100)
    
    if simplang_stats:
        print_stats("SimpleLang+SimpBLAS", simplang_stats)
        
        # Head-to-head comparison
        print("\n" + "="*60)
        print("Head-to-Head Comparison:")
        print("="*60)
        print(f"{'Metric':<25} {'ONNX Runtime':<15} {'SimpleLang':<15} {'Winner':<15}")
        print("-"*70)
        
        metrics = [
            ('Mean Time (ms)', 'mean_time', lambda x: f"{x:.3f}", True),  # Lower is better
            ('Std Dev (ms)', 'std_time', lambda x: f"{x:.3f}", True),
            ('Min Time (ms)', 'min_time', lambda x: f"{x:.3f}", True),
            ('Median Time (ms)', 'median_time', lambda x: f"{x:.3f}", True),
            ('P95 Time (ms)', 'p95_time', lambda x: f"{x:.3f}", True),
            ('P99 Time (ms)', 'p99_time', lambda x: f"{x:.3f}", True),
            ('Throughput (FPS)', 'throughput', lambda x: f"{x:.2f}", False),  # Higher is better
        ]
        
        for metric_name, key, formatter, lower_better in metrics:
            onnx_val = onnx_stats[key]
            simp_val = simplang_stats[key]
            
            if lower_better:
                if onnx_val < simp_val:
                    winner = f"ONNX ({simp_val/onnx_val:.2f}x)"
                else:
                    winner = f"SimpleLang ({onnx_val/simp_val:.2f}x)"
            else:
                if onnx_val > simp_val:
                    winner = f"ONNX ({onnx_val/simp_val:.2f}x)"
                else:
                    winner = f"SimpleLang ({simp_val/onnx_val:.2f}x)"
            
            print(f"{metric_name:<25} {formatter(onnx_val):<15} {formatter(simp_val):<15} {winner:<15}")
        
        print("\n" + "="*60)
        print("Overall Summary:")
        print("="*60)
        
        speedup = onnx_stats['mean_time'] / simplang_stats['mean_time']
        if speedup > 1:
            print(f"✓ SimpleLang+SimpBLAS is {speedup:.2f}x FASTER than ONNX Runtime!")
            print(f"  Our 'shitty implementation' beats ONNX Runtime! 🚀")
        else:
            print(f"✗ ONNX Runtime is {1/speedup:.2f}x faster than SimpleLang+SimpBLAS")
            print(f"  But we're within {100*(1-speedup):.1f}% of ONNX Runtime performance!")
        
        # Save results to JSON for later analysis
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'onnx_runtime': convert_to_serializable(onnx_stats),
            'simplang_simpblas': convert_to_serializable(simplang_stats),
            'speedup': float(speedup)
        }
        
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to benchmark_results.json")
    else:
        print("\nSimpleLang benchmark failed!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "accuracy":
        compare_accuracy()
    else:
        compare_implementations()
        # Also run accuracy comparison after performance
        print("\n\n")
        compare_accuracy()