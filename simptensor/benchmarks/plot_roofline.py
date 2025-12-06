#!/usr/bin/env python3
"""
Roofline Plot for INT8 MatMul Benchmark

Creates a roofline model comparing:
- SimpLang (VNNI-optimized)
- C++ VNNI Reference
- C++ Scalar
- TensorFlow / XLA

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
# Adjust these for your system
PEAK_MEMORY_BW_GB_S = 50.0  # Memory bandwidth in GB/s (DDR4-3200 dual channel ~51 GB/s)
PEAK_INT8_VNNI_GIOPS = 400.0  # Peak INT8 throughput with VNNI (adjust for your CPU)
PEAK_SCALAR_GIOPS = 50.0  # Peak scalar INT8 throughput

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

def plot_roofline(cpp_df, tf_df, output_file='/tmp/int8_roofline.png'):
    """Create roofline plot."""
    fig, ax = plt.subplots(figsize=(14, 9))

    # Arithmetic intensity range
    ai_range = np.logspace(-1, 4, 100)

    # Roofline boundaries
    # Memory-bound region: performance = bandwidth * arithmetic_intensity
    memory_bound = PEAK_MEMORY_BW_GB_S * ai_range

    # Compute-bound region: performance = peak_compute
    vnni_roof = np.full_like(ai_range, PEAK_INT8_VNNI_GIOPS)
    scalar_roof = np.full_like(ai_range, PEAK_SCALAR_GIOPS)

    # Combined roofline (min of memory and compute bounds)
    vnni_roofline = np.minimum(memory_bound, vnni_roof)
    scalar_roofline = np.minimum(memory_bound, scalar_roof)

    # Plot rooflines
    ax.loglog(ai_range, vnni_roofline, 'k-', linewidth=2.5, label=f'VNNI Roofline ({PEAK_INT8_VNNI_GIOPS:.0f} GIOP/s)')
    ax.loglog(ai_range, scalar_roofline, 'k--', linewidth=1.5, alpha=0.6, label=f'Scalar Roofline ({PEAK_SCALAR_GIOPS:.0f} GIOP/s)')

    # Ridge point (where memory and compute bounds meet)
    ridge_ai_vnni = PEAK_INT8_VNNI_GIOPS / PEAK_MEMORY_BW_GB_S
    ax.axvline(x=ridge_ai_vnni, color='gray', linestyle=':', alpha=0.5)
    ax.text(ridge_ai_vnni * 1.1, PEAK_INT8_VNNI_GIOPS * 0.6, f'Ridge\nAI={ridge_ai_vnni:.1f}', fontsize=9, alpha=0.7)

    # Plot benchmark data
    markers = {'simplang': 'o', 'vnni': 's', 'scalar': '^', 'tf': 'D', 'xla': 'p', 'eigen': 'v'}
    colors = {'simplang': '#1f77b4', 'vnni': '#2ca02c', 'scalar': '#d62728', 'tf': '#ff7f0e', 'xla': '#9467bd', 'eigen': '#8c564b'}

    if cpp_df is not None:
        # SimpLang
        ax.scatter(cpp_df['arithmetic_intensity'], cpp_df['simplang_giops'],
                   marker=markers['simplang'], c=colors['simplang'], s=120, zorder=5,
                   label='SimpLang (VNNI)', edgecolors='black', linewidths=0.5)

        # Annotate sizes
        for _, row in cpp_df.iterrows():
            ax.annotate(f"{int(row['N'])}", (row['arithmetic_intensity'], row['simplang_giops']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.8)

        # C++ VNNI Reference
        ax.scatter(cpp_df['arithmetic_intensity'], cpp_df['vnni_giops'],
                   marker=markers['vnni'], c=colors['vnni'], s=90, zorder=4,
                   label='C++ VNNI Reference', edgecolors='black', linewidths=0.5)

        # C++ Scalar
        ax.scatter(cpp_df['arithmetic_intensity'], cpp_df['scalar_giops'],
                   marker=markers['scalar'], c=colors['scalar'], s=70, zorder=3,
                   label='C++ Scalar', edgecolors='black', linewidths=0.5)

        # Eigen
        if 'eigen_giops' in cpp_df.columns:
            ax.scatter(cpp_df['arithmetic_intensity'], cpp_df['eigen_giops'],
                       marker=markers['eigen'], c=colors['eigen'], s=70, zorder=3,
                       label='Eigen', edgecolors='black', linewidths=0.5)

    if tf_df is not None:
        # TensorFlow Eager
        ax.scatter(tf_df['arithmetic_intensity'], tf_df['tf_giops'],
                   marker=markers['tf'], c=colors['tf'], s=90, zorder=4,
                   label='TensorFlow (oneDNN)', edgecolors='black', linewidths=0.5)

        # TensorFlow XLA
        ax.scatter(tf_df['arithmetic_intensity'], tf_df['xla_giops'],
                   marker=markers['xla'], c=colors['xla'], s=90, zorder=4,
                   label='TensorFlow XLA', edgecolors='black', linewidths=0.5)

    # Formatting
    ax.set_xlabel('Arithmetic Intensity (IOPS/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GIOP/s)', fontsize=12)
    ax.set_title('INT8 MatMul Roofline Model\nSimpLang vs C++ VNNI vs Scalar vs NumPy', fontsize=14)

    ax.set_xlim(0.1, 10000)
    ax.set_ylim(0.1, 1000)

    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    # Add efficiency annotations
    if cpp_df is not None and len(cpp_df) > 0:
        max_row = cpp_df.loc[cpp_df['simplang_giops'].idxmax()]
        efficiency = (max_row['simplang_giops'] / PEAK_INT8_VNNI_GIOPS) * 100
        ax.text(0.02, 0.98, f"Peak SimpLang: {max_row['simplang_giops']:.1f} GIOP/s ({efficiency:.1f}% of theoretical peak)",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Roofline plot saved to: {output_file}")

    # Also save as PDF for papers
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
    print("Loading benchmark results...")

    cpp_df = load_cpp_results()
    tf_df = load_tf_results()

    if cpp_df is not None:
        print(f"Loaded {len(cpp_df)} C++ benchmark results")
        print(cpp_df[['N', 'simplang_giops', 'vnni_giops', 'scalar_giops']].to_string())
    else:
        print("No C++ results found. Run: ./bench_int8_matmul_runner_v2 /tmp/bench_int8_matmul.so")

    if tf_df is not None:
        print(f"\nLoaded {len(tf_df)} TensorFlow benchmark results")
        print(tf_df[['N', 'tf_giops', 'xla_giops']].to_string())
    else:
        print("No TensorFlow results found. Run: python3 bench_int8_tflite.py")

    print("\nGenerating plots...")
    plot_roofline(cpp_df, tf_df)
    plot_performance_comparison(cpp_df, tf_df)

    print("\nDone! Plots saved to /tmp/")

if __name__ == '__main__':
    main()
