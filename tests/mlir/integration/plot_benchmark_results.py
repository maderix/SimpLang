#!/usr/bin/env python3
"""
Visualization script for SimpLang vs GCC vs Clang benchmarks
Generates comprehensive performance and bandwidth charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Load data
df = pd.read_csv('/tmp/benchmark_results.csv')

# Calculate speedups
df['Speedup_vs_GCC'] = df['Time_GCC_us'] / df['Time_SimpLang_us']
df['Speedup_vs_Clang'] = df['Time_Clang_us'] / df['Time_SimpLang_us']

print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)
print(f"\nTotal tests: {len(df)}")
print(f"\nSimpLang vs GCC:")
print(f"  Faster: {(df['Speedup_vs_GCC'] > 1.0).sum()} tests")
print(f"  Slower: {(df['Speedup_vs_GCC'] < 1.0).sum()} tests")
print(f"  Average speedup: {df['Speedup_vs_GCC'].mean():.3f}x")
print(f"  Max speedup: {df['Speedup_vs_GCC'].max():.3f}x")
print(f"\nSimpLang vs Clang:")
print(f"  Faster: {(df['Speedup_vs_Clang'] > 1.0).sum()} tests")
print(f"  Slower: {(df['Speedup_vs_Clang'] < 1.0).sum()} tests")
print(f"  Average speedup: {df['Speedup_vs_Clang'].mean():.3f}x")
print(f"  Max speedup: {df['Speedup_vs_Clang'].max():.3f}x")
print()

# ========== CHART 1: Bandwidth Comparison by Datatype ==========
fig, ax = plt.subplots(figsize=(16, 10))

datatypes = df['Datatype'].unique()
n_datatypes = len(datatypes)
x = np.arange(len(df))
width = 0.25

bars1 = ax.bar(x - width, df['BW_SimpLang_GBs'], width, label='SimpLang (MLIR)', color='#2ecc71', alpha=0.9)
bars2 = ax.bar(x, df['BW_GCC_GBs'], width, label='C++ (GCC O3)', color='#3498db', alpha=0.9)
bars3 = ax.bar(x + width, df['BW_Clang_GBs'], width, label='C++ (Clang O3)', color='#e74c3c', alpha=0.9)

ax.set_xlabel('Test', fontweight='bold', fontsize=12)
ax.set_ylabel('Bandwidth (GB/s)', fontweight='bold', fontsize=12)
ax.set_title('Memory Bandwidth Comparison: SimpLang vs GCC vs Clang\n(Higher is Better)',
             fontweight='bold', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f"{row['Datatype']}\n{row['Test']}" for _, row in df.iterrows()],
                     rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=7)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.tight_layout()
plt.savefig('/tmp/benchmark_bandwidth_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /tmp/benchmark_bandwidth_comparison.png")

# ========== CHART 2: Speedup Heatmap ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Reshape data for heatmap
pivot_gcc = df.pivot_table(values='Speedup_vs_GCC', index='Datatype', columns='Test', aggfunc='first')
pivot_clang = df.pivot_table(values='Speedup_vs_Clang', index='Datatype', columns='Test', aggfunc='first')

# Plot heatmaps
sns.heatmap(pivot_gcc, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
            vmin=0.5, vmax=2.0, ax=ax1, cbar_kws={'label': 'Speedup'})
ax1.set_title('SimpLang Speedup vs GCC\n(>1.0 = SimpLang Faster)', fontweight='bold', fontsize=13)
ax1.set_xlabel('Test', fontweight='bold')
ax1.set_ylabel('Datatype', fontweight='bold')

sns.heatmap(pivot_clang, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
            vmin=0.5, vmax=2.0, ax=ax2, cbar_kws={'label': 'Speedup'})
ax2.set_title('SimpLang Speedup vs Clang\n(>1.0 = SimpLang Faster)', fontweight='bold', fontsize=13)
ax2.set_xlabel('Test', fontweight='bold')
ax2.set_ylabel('Datatype', fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/benchmark_speedup_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /tmp/benchmark_speedup_heatmap.png")

# ========== CHART 3: Speedup by Datatype (Box Plot) ==========
fig, ax = plt.subplots(figsize=(14, 8))

speedup_data = []
labels = []
for dtype in datatypes:
    dtype_data = df[df['Datatype'] == dtype]
    speedup_data.append(dtype_data['Speedup_vs_GCC'].values)
    speedup_data.append(dtype_data['Speedup_vs_Clang'].values)
    labels.extend([f'{dtype}\nvs GCC', f'{dtype}\nvs Clang'])

bp = ax.boxplot(speedup_data, labels=labels, patch_artist=True,
                medianprops=dict(color='red', linewidth=2),
                boxprops=dict(facecolor='lightblue', alpha=0.7))

ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (1.0x)')
ax.set_ylabel('Speedup', fontweight='bold', fontsize=12)
ax.set_title('SimpLang Speedup Distribution by Datatype\n(Higher is Better, >1.0 = SimpLang Faster)',
             fontweight='bold', fontsize=14, pad=20)
ax.grid(axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('/tmp/benchmark_speedup_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /tmp/benchmark_speedup_boxplot.png")

# ========== CHART 4: Performance Summary by Operation Type ==========
df['Operation'] = df['Test'].apply(lambda x: x.split('(')[0].strip())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Group by operation type
op_summary = df.groupby('Operation').agg({
    'Speedup_vs_GCC': ['mean', 'min', 'max'],
    'Speedup_vs_Clang': ['mean', 'min', 'max']
})

operations = op_summary.index.tolist()
x_pos = np.arange(len(operations))
width = 0.35

# vs GCC
means_gcc = op_summary['Speedup_vs_GCC']['mean'].values
mins_gcc = op_summary['Speedup_vs_GCC']['min'].values
maxs_gcc = op_summary['Speedup_vs_GCC']['max'].values

bars1 = ax1.bar(x_pos, means_gcc, width, label='Average', color='#2ecc71', alpha=0.8)
ax1.errorbar(x_pos, means_gcc,
             yerr=[means_gcc - mins_gcc, maxs_gcc - means_gcc],
             fmt='none', ecolor='black', capsize=5, capthick=2)
ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0x)')
ax1.set_ylabel('Speedup', fontweight='bold', fontsize=11)
ax1.set_title('SimpLang vs GCC by Operation Type', fontweight='bold', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(operations, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# vs Clang
means_clang = op_summary['Speedup_vs_Clang']['mean'].values
mins_clang = op_summary['Speedup_vs_Clang']['min'].values
maxs_clang = op_summary['Speedup_vs_Clang']['max'].values

bars2 = ax2.bar(x_pos, means_clang, width, label='Average', color='#3498db', alpha=0.8)
ax2.errorbar(x_pos, means_clang,
             yerr=[means_clang - mins_clang, maxs_clang - means_clang],
             fmt='none', ecolor='black', capsize=5, capthick=2)
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0x)')
ax2.set_ylabel('Speedup', fontweight='bold', fontsize=11)
ax2.set_title('SimpLang vs Clang by Operation Type', fontweight='bold', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(operations, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/benchmark_operation_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /tmp/benchmark_operation_summary.png")

# ========== CHART 5: Peak Bandwidth by Datatype ==========
fig, ax = plt.subplots(figsize=(14, 8))

peak_bw = df.groupby('Datatype').agg({
    'BW_SimpLang_GBs': 'max',
    'BW_GCC_GBs': 'max',
    'BW_Clang_GBs': 'max'
})

x = np.arange(len(peak_bw))
width = 0.25

ax.bar(x - width, peak_bw['BW_SimpLang_GBs'], width, label='SimpLang (MLIR)', color='#2ecc71')
ax.bar(x, peak_bw['BW_GCC_GBs'], width, label='C++ (GCC O3)', color='#3498db')
ax.bar(x + width, peak_bw['BW_Clang_GBs'], width, label='C++ (Clang O3)', color='#e74c3c')

ax.set_ylabel('Peak Bandwidth (GB/s)', fontweight='bold', fontsize=12)
ax.set_title('Peak Memory Bandwidth by Datatype\n(Higher is Better)', fontweight='bold', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(peak_bw.index, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, dtype in enumerate(peak_bw.index):
    for j, (col, offset) in enumerate([('BW_SimpLang_GBs', -width),
                                        ('BW_GCC_GBs', 0),
                                        ('BW_Clang_GBs', width)]):
        val = peak_bw.loc[dtype, col]
        ax.text(i + offset, val, f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/benchmark_peak_bandwidth.png', dpi=300, bbox_inches='tight')
print("✓ Saved: /tmp/benchmark_peak_bandwidth.png")

# ========== Generate Summary Report ==========
with open('/tmp/benchmark_summary.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SIMPLANG COMPREHENSIVE BENCHMARK REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total Tests: {len(df)}\n\n")

    f.write("OVERALL PERFORMANCE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"SimpLang vs GCC:\n")
    f.write(f"  Average Speedup: {df['Speedup_vs_GCC'].mean():.3f}x\n")
    f.write(f"  Geometric Mean: {np.exp(np.log(df['Speedup_vs_GCC']).mean()):.3f}x\n")
    f.write(f"  Tests Faster: {(df['Speedup_vs_GCC'] > 1.0).sum()}/{len(df)} ({100*(df['Speedup_vs_GCC'] > 1.0).sum()/len(df):.1f}%)\n")
    f.write(f"  Best Speedup: {df['Speedup_vs_GCC'].max():.3f}x ({df.loc[df['Speedup_vs_GCC'].idxmax(), 'Datatype']} - {df.loc[df['Speedup_vs_GCC'].idxmax(), 'Test']})\n\n")

    f.write(f"SimpLang vs Clang:\n")
    f.write(f"  Average Speedup: {df['Speedup_vs_Clang'].mean():.3f}x\n")
    f.write(f"  Geometric Mean: {np.exp(np.log(df['Speedup_vs_Clang']).mean()):.3f}x\n")
    f.write(f"  Tests Faster: {(df['Speedup_vs_Clang'] > 1.0).sum()}/{len(df)} ({100*(df['Speedup_vs_Clang'] > 1.0).sum()/len(df):.1f}%)\n")
    f.write(f"  Best Speedup: {df['Speedup_vs_Clang'].max():.3f}x ({df.loc[df['Speedup_vs_Clang'].idxmax(), 'Datatype']} - {df.loc[df['Speedup_vs_Clang'].idxmax(), 'Test']})\n\n")

    f.write("\nTOP 5 PERFORMANCE WINS (vs GCC):\n")
    f.write("-" * 80 + "\n")
    top5_gcc = df.nlargest(5, 'Speedup_vs_GCC')[['Datatype', 'Test', 'Speedup_vs_GCC', 'BW_SimpLang_GBs', 'BW_GCC_GBs']]
    for idx, row in top5_gcc.iterrows():
        f.write(f"{row['Datatype']:6s} - {row['Test']:30s}: {row['Speedup_vs_GCC']:5.2f}x  ({row['BW_SimpLang_GBs']:6.1f} vs {row['BW_GCC_GBs']:6.1f} GB/s)\n")

    f.write("\nTOP 5 PERFORMANCE WINS (vs Clang):\n")
    f.write("-" * 80 + "\n")
    top5_clang = df.nlargest(5, 'Speedup_vs_Clang')[['Datatype', 'Test', 'Speedup_vs_Clang', 'BW_SimpLang_GBs', 'BW_Clang_GBs']]
    for idx, row in top5_clang.iterrows():
        f.write(f"{row['Datatype']:6s} - {row['Test']:30s}: {row['Speedup_vs_Clang']:5.2f}x  ({row['BW_SimpLang_GBs']:6.1f} vs {row['BW_Clang_GBs']:6.1f} GB/s)\n")

    f.write("\nPEAK BANDWIDTH BY DATATYPE:\n")
    f.write("-" * 80 + "\n")
    for dtype in datatypes:
        dtype_data = df[df['Datatype'] == dtype]
        max_sl = dtype_data['BW_SimpLang_GBs'].max()
        max_gcc = dtype_data['BW_GCC_GBs'].max()
        max_clang = dtype_data['BW_Clang_GBs'].max()
        f.write(f"{dtype:6s}: SimpLang={max_sl:6.1f} GB/s  |  GCC={max_gcc:6.1f} GB/s  |  Clang={max_clang:6.1f} GB/s\n")

print("✓ Saved: /tmp/benchmark_summary.txt")
print("\n" + "=" * 70)
print("Visualization complete! Generated files:")
print("  - /tmp/benchmark_bandwidth_comparison.png")
print("  - /tmp/benchmark_speedup_heatmap.png")
print("  - /tmp/benchmark_speedup_boxplot.png")
print("  - /tmp/benchmark_operation_summary.png")
print("  - /tmp/benchmark_peak_bandwidth.png")
print("  - /tmp/benchmark_summary.txt")
print("=" * 70)
