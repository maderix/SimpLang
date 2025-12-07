#!/usr/bin/env python3
"""
Roofline Plot for INT8 MatMul Benchmark

Creates a roofline model comparing:
- SimpLang VNNI (single-threaded and parallel)
- C++ VNNI Reference
- TFLite XNNPACK
- Eigen

Usage:
    python3 plot_roofline.py

Prerequisites:
    pip install matplotlib pandas
"""

import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# System parameters (AMD Zen3 / Intel with AVX512-VNNI)
PEAK_MEMORY_BW_GB_S = 50.0  # Memory bandwidth in GB/s (DDR4-3200 dual channel ~51 GB/s)
PEAK_INT8_VNNI_GIOPS = 2800.0  # Peak INT8 throughput with VNNI 8-thread (adjust for your CPU)
PEAK_INT8_VNNI_1T_GIOPS = 350.0  # Peak INT8 single-thread
PEAK_SCALAR_GIOPS = 50.0  # Peak scalar INT8 throughput

# Hardcoded benchmark results from our testing
PARALLEL_BENCHMARK_DATA = {
    # Size: (1T, 2T, 4T, 8T) GIOP/s
    256:  (118, 230, 453, 730),
    512:  (188, 370, 685, 1316),
    768:  (224, 410, 817, 1337),
    1024: (244, 482, 575, 1760),
    2048: (280, 563, 963, 1829),
}

# Reference implementations at 1024x1024 (8 threads)
REFERENCE_DATA = {
    'tflite_xnnpack': 1327,  # GIOP/s
    'cpp_vnni_8t': 2534,     # GIOP/s
}

def load_cpp_results(filepath='/tmp/int8_matmul_benchmark.csv'):
    """Load C++ benchmark results."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Run bench_int8_matmul_runner_v2 first.")
        return None
    return pd.read_csv(filepath)

def load_tf_results(filepath='/tmp/int8_tflite_benchmark.csv'):
    """Load TensorFlow/XLA benchmark results."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Run bench_int8_tflite.py first.")
        return None
    return pd.read_csv(filepath)

def plot_thread_scaling(output_file='/tmp/int8_thread_scaling.png'):
    """Create thread scaling line plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sizes = list(PARALLEL_BENCHMARK_DATA.keys())
    threads = [1, 2, 4, 8]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    # Plot 1: Performance vs Threads for each size
    for i, size in enumerate(sizes):
        perf = PARALLEL_BENCHMARK_DATA[size]
        ax1.plot(threads, perf, marker=markers[i], color=colors[i],
                linewidth=2, markersize=8, label=f'{size}×{size}')

    # Add reference lines
    ax1.axhline(y=REFERENCE_DATA['tflite_xnnpack'], color='gray', linestyle='--',
                linewidth=2, alpha=0.7, label=f"TFLite XNNPACK ({REFERENCE_DATA['tflite_xnnpack']})")
    ax1.axhline(y=REFERENCE_DATA['cpp_vnni_8t'], color='black', linestyle=':',
                linewidth=2, alpha=0.7, label=f"C++ VNNI 8T ({REFERENCE_DATA['cpp_vnni_8t']})")

    ax1.set_xlabel('Threads', fontsize=12)
    ax1.set_ylabel('Performance (GIOP/s)', fontsize=12)
    ax1.set_title('SimpLang INT8 VNNI Thread Scaling', fontsize=13)
    ax1.set_xticks(threads)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 2800)

    # Plot 2: Performance vs Size for each thread count
    for i, t in enumerate(threads):
        perf = [PARALLEL_BENCHMARK_DATA[s][i] for s in sizes]
        ax2.plot(sizes, perf, marker=markers[i], color=colors[i],
                linewidth=2, markersize=8, label=f'{t} thread{"s" if t > 1 else ""}')

    ax2.axhline(y=REFERENCE_DATA['tflite_xnnpack'], color='gray', linestyle='--',
                linewidth=2, alpha=0.7, label='TFLite XNNPACK')

    ax2.set_xlabel('Matrix Size (N×N)', fontsize=12)
    ax2.set_ylabel('Performance (GIOP/s)', fontsize=12)
    ax2.set_title('SimpLang INT8 VNNI vs Matrix Size', fontsize=13)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Thread scaling plot saved to: {output_file}")

    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF saved to: {pdf_file}")
    plt.close()


def plot_comparison_bars(output_file='/tmp/int8_comparison.png'):
    """Create comparison bar chart at 1024x1024."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data for 1024x1024
    implementations = ['SimpLang\n1T', 'SimpLang\n2T', 'SimpLang\n4T', 'SimpLang\n8T',
                       'TFLite\nXNNPACK', 'C++ VNNI\n8T']
    values = list(PARALLEL_BENCHMARK_DATA[1024]) + [REFERENCE_DATA['tflite_xnnpack'],
                                                     REFERENCE_DATA['cpp_vnni_8t']]
    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e', '#2ca02c']
    alphas = [0.4, 0.6, 0.8, 1.0, 0.9, 0.9]

    # Create bars individually to support per-bar alpha
    x = np.arange(len(implementations))
    for i, (impl, val, col, alpha) in enumerate(zip(implementations, values, colors, alphas)):
        ax.bar(x[i], val, color=col, alpha=alpha, edgecolor='black', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(implementations)
    bars = ax.patches  # Get bars for annotation

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Performance (GIOP/s)', fontsize=12)
    ax.set_title('INT8 MatMul Performance Comparison (1024×1024)', fontsize=14)
    ax.set_ylim(0, 2800)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax.annotate('SimpLang 8T beats\nTFLite by 33%', xy=(3, 1760), xytext=(3.5, 2200),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")

    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()


def plot_roofline(cpp_df, tf_df, output_file='/tmp/int8_roofline.png'):
    """Create roofline plot with lines."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Arithmetic intensity for INT8 matmul: N^3 ops / (2*N^2 bytes) = N/2
    sizes = list(PARALLEL_BENCHMARK_DATA.keys())
    ai_values = [s / 2 for s in sizes]  # Arithmetic intensity

    # Roofline boundaries
    ai_range = np.logspace(1, 4, 100)
    memory_bound = PEAK_MEMORY_BW_GB_S * ai_range
    vnni_roof_8t = np.full_like(ai_range, PEAK_INT8_VNNI_GIOPS)
    vnni_roof_1t = np.full_like(ai_range, PEAK_INT8_VNNI_1T_GIOPS)

    vnni_roofline_8t = np.minimum(memory_bound, vnni_roof_8t)
    vnni_roofline_1t = np.minimum(memory_bound, vnni_roof_1t)

    # Plot rooflines
    ax.loglog(ai_range, vnni_roofline_8t, 'k-', linewidth=2,
              label=f'8T Roofline ({PEAK_INT8_VNNI_GIOPS:.0f} GIOP/s)', alpha=0.7)
    ax.loglog(ai_range, vnni_roofline_1t, 'k--', linewidth=1.5,
              label=f'1T Roofline ({PEAK_INT8_VNNI_1T_GIOPS:.0f} GIOP/s)', alpha=0.5)

    # Plot SimpLang performance as lines
    colors = {'1T': '#1f77b4', '2T': '#ff7f0e', '4T': '#2ca02c', '8T': '#d62728'}

    for i, (label, color) in enumerate(colors.items()):
        perf = [PARALLEL_BENCHMARK_DATA[s][i] for s in sizes]
        ax.loglog(ai_values, perf, marker='o', color=color, linewidth=2.5,
                  markersize=10, label=f'SimpLang {label}')

    # Add reference point for TFLite at 1024
    ax.scatter([512], [REFERENCE_DATA['tflite_xnnpack']], marker='s', c='gray',
               s=150, zorder=5, label=f"TFLite XNNPACK ({REFERENCE_DATA['tflite_xnnpack']})",
               edgecolors='black', linewidths=1)

    # Add reference point for C++ VNNI at 1024
    ax.scatter([512], [REFERENCE_DATA['cpp_vnni_8t']], marker='^', c='black',
               s=150, zorder=5, label=f"C++ VNNI 8T ({REFERENCE_DATA['cpp_vnni_8t']})",
               edgecolors='black', linewidths=1)

    # Formatting
    ax.set_xlabel('Arithmetic Intensity (IOPS/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GIOP/s)', fontsize=12)
    ax.set_title('INT8 MatMul Roofline Model\nSimpLang VNNI Thread Scaling', fontsize=14)

    ax.set_xlim(50, 2000)
    ax.set_ylim(50, 3000)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    # Add size annotations
    for s, ai in zip(sizes, ai_values):
        ax.annotate(f'{s}', (ai, PARALLEL_BENCHMARK_DATA[s][3] * 1.1),
                    fontsize=9, ha='center', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Roofline plot saved to: {output_file}")

    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF saved to: {pdf_file}")
    plt.close()

def plot_performance_comparison(cpp_df, tf_df, output_file='/tmp/int8_performance.png'):
    """Create performance comparison bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    if cpp_df is None:
        print("No C++ data available for performance comparison")
        return

    sizes = cpp_df['N'].astype(str)
    x = np.arange(len(sizes))
    width = 0.15

    # GIOP/s comparison
    ax1.bar(x - 2*width, cpp_df['simplang_giops'], width, label='SimpLang', color='#1f77b4', alpha=0.9)
    ax1.bar(x - 1*width, cpp_df['vnni_giops'], width, label='C++ VNNI', color='#2ca02c', alpha=0.9)
    ax1.bar(x, cpp_df['scalar_giops'], width, label='Scalar', color='#d62728', alpha=0.9)
    if tf_df is not None:
        tf_giops = tf_df.set_index('N').reindex(cpp_df['N'])['tf_giops'].values
        xla_giops = tf_df.set_index('N').reindex(cpp_df['N'])['xla_giops'].values
        ax1.bar(x + 1*width, tf_giops, width, label='TF oneDNN', color='#ff7f0e', alpha=0.9)
        ax1.bar(x + 2*width, xla_giops, width, label='TF XLA', color='#9467bd', alpha=0.9)

    ax1.set_ylabel('GIOP/s', fontsize=11)
    ax1.set_xlabel('Matrix Size (N×N)', fontsize=11)
    ax1.set_title('INT8 MatMul Performance Comparison', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes, rotation=45)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')

    # Speedup vs scalar
    speedup_simplang = cpp_df['simplang_giops'] / cpp_df['scalar_giops']
    speedup_vnni = cpp_df['vnni_giops'] / cpp_df['scalar_giops']

    ax2.bar(x - 0.2, speedup_simplang, 0.35, label='SimpLang vs Scalar', color='#1f77b4', alpha=0.9)
    ax2.bar(x + 0.2, speedup_vnni, 0.35, label='C++ VNNI vs Scalar', color='#2ca02c', alpha=0.9)

    ax2.set_ylabel('Speedup (×)', fontsize=11)
    ax2.set_xlabel('Matrix Size (N×N)', fontsize=11)
    ax2.set_title('Speedup over Scalar Reference', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes, rotation=45)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Performance comparison saved to: {output_file}")
    plt.close()

def main():
    print("=" * 60)
    print("INT8 VNNI Benchmark Visualization")
    print("=" * 60)

    # Print hardcoded benchmark data
    print("\nParallel Benchmark Data (GIOP/s):")
    print("-" * 50)
    print(f"{'Size':<10} {'1T':<8} {'2T':<8} {'4T':<8} {'8T':<8}")
    print("-" * 50)
    for size, perf in PARALLEL_BENCHMARK_DATA.items():
        print(f"{size:<10} {perf[0]:<8} {perf[1]:<8} {perf[2]:<8} {perf[3]:<8}")
    print("-" * 50)
    print(f"\nReference (1024x1024, 8T):")
    print(f"  TFLite XNNPACK: {REFERENCE_DATA['tflite_xnnpack']} GIOP/s")
    print(f"  C++ VNNI:       {REFERENCE_DATA['cpp_vnni_8t']} GIOP/s")

    # Generate plots using hardcoded data
    print("\nGenerating plots...")

    # New line plots
    plot_thread_scaling('/tmp/int8_thread_scaling.png')
    plot_comparison_bars('/tmp/int8_comparison.png')

    # Roofline (uses hardcoded data now)
    plot_roofline(None, None, '/tmp/int8_roofline.png')

    # Try to load CSV data for legacy plots
    cpp_df = load_cpp_results()
    tf_df = load_tf_results()
    if cpp_df is not None:
        plot_performance_comparison(cpp_df, tf_df, '/tmp/int8_performance.png')

    print("\n" + "=" * 60)
    print("Plots saved to /tmp/:")
    print("  - int8_thread_scaling.png/pdf  (NEW: line plots)")
    print("  - int8_comparison.png/pdf      (NEW: bar comparison)")
    print("  - int8_roofline.png/pdf        (roofline model)")
    print("=" * 60)

if __name__ == '__main__':
    main()
