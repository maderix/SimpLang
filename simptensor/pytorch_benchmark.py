#!/usr/bin/env python3
"""
PyTorch CPU benchmark comparison with SimpTensor
Tests similar operations to tensor_perf.sl for fair comparison
"""

import torch
import time
import numpy as np

def pytorch_benchmark():
    print("=== PyTorch CPU Benchmark ===")
    
    # Same tensor dimensions as SimpTensor test
    batch_size = 4
    height = 64
    width = 64
    channels = 128
    total_elements = batch_size * height * width * channels
    
    print(f"Tensor dimensions: {batch_size}x{height}x{width}x{channels} = {total_elements} elements")
    print(f"Memory usage: ~{total_elements * 4 / (1024 * 1024):.0f} MB")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MKLDNN enabled: {torch.backends.mkldnn.is_available()}")
    print(f"OpenMP threads: {torch.get_num_threads()}")
    print()
    
    # Create tensor with NHWC layout (same as SimpTensor)
    tensor_data = torch.zeros(batch_size, height, width, channels, dtype=torch.float32)
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        # Initialization
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        tensor_data[b, h, w, c] = b * 1000.0 + h * 100.0 + w * 10.0 + c * 1.0
        
        # Element-wise operations
        computed = tensor_data * tensor_data * 0.001 + tensor_data * 0.1 + 1.0
        computed = computed * computed * 0.5
        compute_sum = computed.sum()
        
        # Strided access (channel 0 across spatial dims)
        stride_sum = tensor_data[:, :, :, 0].sum()
        
        # Spatial reduction
        reduced = tensor_data.mean(dim=(1, 2))  # Reduce over height and width
        reduce_sum = (reduced * reduced).sum()
    
    # Performance measurement
    num_runs = 10
    times = []
    results = []
    
    print(f"Running {num_runs} performance iterations...")
    
    for run in range(num_runs):
        start_time = time.perf_counter()
        
        # Test 1: Tensor initialization (write stress test)
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        tensor_data[b, h, w, c] = b * 1000.0 + h * 100.0 + w * 10.0 + c * 1.0
        
        # Test 2: Element-wise operations (computation stress test)
        computed = tensor_data * tensor_data * 0.001 + tensor_data * 0.1 + 1.0
        computed = computed * computed * 0.5
        compute_sum = computed.sum()
        compute_count = total_elements
        
        # Test 3: Strided access patterns (channel 0 across spatial dims)
        stride_sum = tensor_data[:, :, :, 0].sum()
        stride_ops = batch_size * height * width
        
        # Test 4: Spatial reduction operations
        reduced = tensor_data.mean(dim=(1, 2))  # Reduce over height and width
        reduce_sum = (reduced * reduced).sum()
        reduce_count = batch_size * channels
        
        # Final calculation (matching SimpTensor logic)
        avg_compute = compute_sum / compute_count
        avg_stride = stride_sum / stride_ops  
        avg_reduce = reduce_sum / reduce_count
        final_result = avg_compute * 0.001 + avg_stride * 0.0001 + avg_reduce * 0.00001
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        times.append(elapsed_ms)
        results.append(float(final_result))
        
        print(f"Run {run+1}: {elapsed_ms:.4f} ms (result: {final_result:.0f})")
    
    # Performance analysis
    avg_time = np.mean(times)
    best_time = np.min(times)
    worst_time = np.max(times)
    
    print()
    print("=== PyTorch Performance Results ===")
    print(f"Average time: {avg_time:.4f} ms")
    print(f"Best time:    {best_time:.4f} ms") 
    print(f"Worst time:   {worst_time:.4f} ms")
    print(f"Total elements processed: {total_elements}")
    
    # Calculate throughput metrics
    operations_per_run = total_elements * 4  # 4 major operation phases
    throughput_best = (operations_per_run / (best_time / 1000.0)) / 1e6  # M ops/sec
    throughput_avg = (operations_per_run / (avg_time / 1000.0)) / 1e6   # M ops/sec
    
    print(f"Peak throughput:    {throughput_best:.3f} M ops/sec")
    print(f"Average throughput: {throughput_avg:.3f} M ops/sec")
    
    # Memory bandwidth estimation
    memory_ops = total_elements * 8  # Rough estimate: read + write per element
    memory_bandwidth = (memory_ops * 4 / (best_time / 1000.0)) / (1024 * 1024 * 1024)  # GB/sec
    print(f"Est. memory bandwidth: {memory_bandwidth:.5f} GB/sec")
    
    return {
        'avg_time': avg_time,
        'best_time': best_time,
        'worst_time': worst_time,
        'peak_throughput': throughput_best,
        'avg_throughput': throughput_avg,
        'memory_bandwidth': memory_bandwidth
    }

def pytorch_vectorized_benchmark():
    """Optimized PyTorch version using vectorized operations"""
    print("\n=== PyTorch Vectorized Benchmark ===")
    
    batch_size = 4
    height = 64
    width = 64
    channels = 128
    total_elements = batch_size * height * width * channels
    
    print(f"Using vectorized PyTorch operations for better performance")
    print()
    
    # Pre-allocate tensors
    tensor_data = torch.zeros(batch_size, height, width, channels, dtype=torch.float32)
    
    # Create initialization tensor using broadcasting
    b_vals = torch.arange(batch_size).view(-1, 1, 1, 1) * 1000.0
    h_vals = torch.arange(height).view(1, -1, 1, 1) * 100.0  
    w_vals = torch.arange(width).view(1, 1, -1, 1) * 10.0
    c_vals = torch.arange(channels).view(1, 1, 1, -1) * 1.0
    init_pattern = b_vals + h_vals + w_vals + c_vals
    
    # Warmup
    print("Warming up vectorized version...")
    for _ in range(3):
        tensor_data.copy_(init_pattern)
        computed = tensor_data * tensor_data * 0.001 + tensor_data * 0.1 + 1.0
        computed = computed * computed * 0.5
        compute_sum = computed.sum()
        stride_sum = tensor_data[:, :, :, 0].sum()
        reduced = tensor_data.mean(dim=(1, 2))
        reduce_sum = (reduced * reduced).sum()
    
    # Performance measurement
    num_runs = 10
    times = []
    
    print(f"Running {num_runs} vectorized performance iterations...")
    
    for run in range(num_runs):
        start_time = time.perf_counter()
        
        # Test 1: Vectorized initialization
        tensor_data.copy_(init_pattern)
        
        # Test 2: Vectorized element-wise operations
        computed = tensor_data * tensor_data * 0.001 + tensor_data * 0.1 + 1.0
        computed = computed * computed * 0.5
        compute_sum = computed.sum()
        compute_count = total_elements
        
        # Test 3: Vectorized strided access
        stride_sum = tensor_data[:, :, :, 0].sum()
        stride_ops = batch_size * height * width
        
        # Test 4: Vectorized spatial reduction
        reduced = tensor_data.mean(dim=(1, 2))
        reduce_sum = (reduced * reduced).sum()
        reduce_count = batch_size * channels
        
        # Final calculation
        avg_compute = compute_sum / compute_count
        avg_stride = stride_sum / stride_ops
        avg_reduce = reduce_sum / reduce_count
        final_result = avg_compute * 0.001 + avg_stride * 0.0001 + avg_reduce * 0.00001
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        times.append(elapsed_ms)
        print(f"Run {run+1}: {elapsed_ms:.4f} ms (result: {final_result:.0f})")
    
    # Performance analysis
    avg_time = np.mean(times)
    best_time = np.min(times)
    worst_time = np.max(times)
    
    print()
    print("=== PyTorch Vectorized Performance Results ===")
    print(f"Average time: {avg_time:.4f} ms")
    print(f"Best time:    {best_time:.4f} ms")
    print(f"Worst time:   {worst_time:.4f} ms")
    
    operations_per_run = total_elements * 4
    throughput_best = (operations_per_run / (best_time / 1000.0)) / 1e6
    throughput_avg = (operations_per_run / (avg_time / 1000.0)) / 1e6
    
    print(f"Peak throughput:    {throughput_best:.3f} M ops/sec")
    print(f"Average throughput: {throughput_avg:.3f} M ops/sec")
    
    memory_ops = total_elements * 8
    memory_bandwidth = (memory_ops * 4 / (best_time / 1000.0)) / (1024 * 1024 * 1024)
    print(f"Est. memory bandwidth: {memory_bandwidth:.5f} GB/sec")
    
    return {
        'avg_time': avg_time,
        'best_time': best_time,
        'worst_time': worst_time,
        'peak_throughput': throughput_best,
        'avg_throughput': throughput_avg,
        'memory_bandwidth': memory_bandwidth
    }

def compare_results():
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    # SimpTensor results (from our previous run)
    simptensor = {
        'avg_time': 17.8112,
        'best_time': 17.6618, 
        'worst_time': 18.3273,
        'peak_throughput': 474.958,
        'avg_throughput': 470.975,
        'memory_bandwidth': 3.53872
    }
    
    # Run PyTorch benchmarks
    pytorch_naive = pytorch_benchmark()
    pytorch_opt = pytorch_vectorized_benchmark()
    
    print(f"\n{'Metric':<25} {'SimpTensor':<15} {'PyTorch (naive)':<18} {'PyTorch (opt)':<15} {'Best':<10}")
    print("-" * 85)
    
    metrics = [
        ('Best time (ms)', 'best_time', 'lower'),
        ('Avg time (ms)', 'avg_time', 'lower'), 
        ('Peak throughput (M/s)', 'peak_throughput', 'higher'),
        ('Avg throughput (M/s)', 'avg_throughput', 'higher'),
        ('Memory BW (GB/s)', 'memory_bandwidth', 'higher')
    ]
    
    for name, key, direction in metrics:
        st_val = simptensor[key]
        pt_naive_val = pytorch_naive[key]
        pt_opt_val = pytorch_opt[key]
        
        if direction == 'lower':
            best_val = min(st_val, pt_naive_val, pt_opt_val)
        else:
            best_val = max(st_val, pt_naive_val, pt_opt_val)
            
        best_name = ""
        if abs(st_val - best_val) < 0.001:
            best_name = "SimpTensor"
        elif abs(pt_opt_val - best_val) < 0.001:
            best_name = "PyTorch"
        elif abs(pt_naive_val - best_val) < 0.001:
            best_name = "PyTorch"
            
        print(f"{name:<25} {st_val:<15.3f} {pt_naive_val:<18.3f} {pt_opt_val:<15.3f} {best_name:<10}")
    
    # Calculate speedups
    print(f"\nSpeedup Analysis:")
    print(f"SimpTensor vs PyTorch (naive):     {pytorch_naive['best_time'] / simptensor['best_time']:.2f}x faster")
    print(f"SimpTensor vs PyTorch (optimized): {pytorch_opt['best_time'] / simptensor['best_time']:.2f}x faster")
    print(f"PyTorch (opt) vs PyTorch (naive):  {pytorch_naive['best_time'] / pytorch_opt['best_time']:.2f}x faster")

if __name__ == "__main__":
    compare_results()