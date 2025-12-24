#!/usr/bin/env python3
"""
Plot VNNI INT8 matmul sweep results.
Generates visualization of performance across matrix sizes and tile configurations.
Supports both square tile sweep and M,N,K independent sweep modes.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_mnk_sweep(input_csv='/tmp/mnk_sweep_results.csv', output_png='/tmp/mnk_sweep_plot.png'):
    """Plot M,N,K sweep results as 3-way heatmaps."""
    df = pd.read_csv(input_csv)

    # Platform info
    platform = "AMD Ryzen 7 7800X3D"
    adjusted_peak = 2560

    # Get unique values
    matrix_sizes = sorted(df['N'].unique())
    k_tiles = sorted(df['K_tile'].unique())
    m_tiles = sorted(df['M_tile'].unique())
    n_tiles = sorted(df['N_tile'].unique())

    # Create figure: one row per matrix size, one column per K tile
    n_rows = len(matrix_sizes)
    n_cols = len(k_tiles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 2, 4 * n_rows + 1))
    fig.suptitle(f'SimpLang VNNI INT8 MatMul: M×N Tile Performance by K Tile\n{platform} | 8C @ 5GHz | Peak: {adjusted_peak} GIOP/s',
                 fontsize=14, fontweight='bold', y=0.98)

    # Handle single row/col case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Global min/max for consistent colorbar
    vmin = df['GIOPS'].min()
    vmax = df['GIOPS'].max()

    for row_idx, mat_size in enumerate(matrix_sizes):
        for col_idx, k_tile in enumerate(k_tiles):
            ax = axes[row_idx, col_idx]

            # Filter data for this matrix size and K tile
            subset = df[(df['N'] == mat_size) & (df['K_tile'] == k_tile)]

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'N={mat_size}, K={k_tile}', fontsize=10)
                continue

            # Create pivot table: M_tile vs N_tile
            pivot = subset.pivot(index='M_tile', columns='N_tile', values='GIOPS')

            # Plot heatmap
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)

            # Labels
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)

            # Title with peak for this config
            max_giops = pivot.values[~np.isnan(pivot.values)].max() if not np.all(np.isnan(pivot.values)) else 0
            pct_peak = 100 * max_giops / adjusted_peak
            ax.set_title(f'N={mat_size}, K={k_tile}\nPeak: {max_giops:.0f} ({pct_peak:.0f}%)', fontsize=10)

            if row_idx == n_rows - 1:
                ax.set_xlabel('N Tile', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel('M Tile', fontsize=10)

            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        color = 'white' if val > vmax * 0.6 else 'black'
                        ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7, color=color)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('GIOP/s', fontsize=11)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {output_png}")

    # Print analysis
    print("\n=== M,N,K Sweep Analysis ===")
    print(f"Peak observed: {df['GIOPS'].max():.0f} GIOP/s ({100*df['GIOPS'].max()/adjusted_peak:.1f}%)")

    # Best config per matrix size
    print("\nBest configuration per matrix size:")
    for mat_size in matrix_sizes:
        subset = df[df['N'] == mat_size]
        best = subset.loc[subset['GIOPS'].idxmax()]
        print(f"  N={mat_size}: M={int(best['M_tile'])}, N={int(best['N_tile'])}, K={int(best['K_tile'])} -> {best['GIOPS']:.0f} GIOP/s")

    # K tile analysis
    print("\nBest K tile by matrix size:")
    for mat_size in matrix_sizes:
        subset = df[df['N'] == mat_size]
        k_perf = subset.groupby('K_tile')['GIOPS'].max()
        best_k = k_perf.idxmax()
        print(f"  N={mat_size}: K={best_k} (max {k_perf[best_k]:.0f} GIOP/s)")

    return df


def plot_sweep(input_csv='/tmp/sweep_results.csv', output_png='/tmp/vnni_sweep_plot.png'):
    # Read data
    df = pd.read_csv(input_csv)

    # Platform info
    platform = "AMD Ryzen 7 7800X3D"
    cores = 8
    freq = 5.0  # GHz
    arch = "Zen 4 (AVX-VNNI)"
    l3_cache = "96MB (3D V-Cache)"

    # Theoretical peak: Zen 4 has 2 VNNI units per core = 2560 GIOP/s
    adjusted_peak = 2560

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'SimpLang VNNI INT8 MatMul Performance (tensor_matmul_out)\n{platform} | {cores}C @ {freq}GHz | {arch} | {l3_cache}',
                 fontsize=14, fontweight='bold')

    # Color scheme for tiles
    colors = {32: '#e74c3c', 64: '#3498db', 128: '#2ecc71', 256: '#9b59b6'}
    markers = {32: 'o', 64: 's', 128: '^', 256: 'D'}

    # Plot 1: GIOP/s vs Matrix Size (grouped by tile) - log scale for x-axis
    ax1 = axes[0, 0]
    for tile in sorted(df['Tile'].unique()):
        if tile in colors:
            subset = df[df['Tile'] == tile].sort_values('N')
            ax1.plot(subset['N'], subset['GIOPS'], marker=markers.get(tile, 'o'),
                     color=colors.get(tile, 'gray'), linewidth=2, markersize=6,
                     label=f'Tile {tile}x{tile}x16')
    ax1.axhline(y=adjusted_peak, color='red', linestyle=':', alpha=0.5, label=f'Peak ({adjusted_peak} GIOP/s)')
    ax1.set_xlabel('Matrix Size (N×N)', fontsize=11)
    ax1.set_ylabel('Performance (GIOP/s)', fontsize=11)
    ax1.set_title('Performance vs Matrix Size by Tile Configuration', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_ylim(0, max(df['GIOPS'].max() * 1.1, adjusted_peak * 0.9))

    # Plot 2: Heatmap of performance
    ax2 = axes[0, 1]
    pivot = df.pivot(index='Tile', columns='N', values='GIOPS')
    im = ax2.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    ax2.set_xticks(range(len(pivot.columns)))
    ax2.set_xticklabels([f'{n//1024}K' if n >= 1024 else str(n) for n in pivot.columns], rotation=45, fontsize=8)
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_yticklabels([f'{t}x{t}' for t in pivot.index])
    ax2.set_xlabel('Matrix Size (N)', fontsize=11)
    ax2.set_ylabel('Tile Size', fontsize=11)
    ax2.set_title('Performance Heatmap (GIOP/s)', fontsize=12)
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('GIOP/s')
    # Add text annotations (skip if too many columns)
    if len(pivot.columns) <= 12:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax2.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7,
                            color='white' if val > pivot.values[~np.isnan(pivot.values)].max()*0.6 else 'black')

    # Plot 3: Best tile per size
    ax3 = axes[1, 0]
    best_per_size = df.loc[df.groupby('N')['GIOPS'].idxmax()]
    bar_colors = [colors.get(t, 'gray') for t in best_per_size['Tile']]
    bars = ax3.bar(range(len(best_per_size)), best_per_size['GIOPS'],
                   color=bar_colors, edgecolor='black')
    ax3.set_xticks(range(len(best_per_size)))
    ax3.set_xticklabels([f'{n//1024}K' if n >= 1024 else str(n) for n in best_per_size['N']], rotation=45, fontsize=9)
    ax3.set_xlabel('Matrix Size (N)', fontsize=11)
    ax3.set_ylabel('Best Performance (GIOP/s)', fontsize=11)
    ax3.set_title('Optimal Performance per Matrix Size', fontsize=12)
    ax3.axhline(y=adjusted_peak, color='red', linestyle=':', alpha=0.5)
    # Add tile size labels on bars
    for i, (idx, row) in enumerate(best_per_size.iterrows()):
        ax3.text(i, row['GIOPS'] + 30, f'{int(row["Tile"])}²', ha='center', fontsize=8, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: % of peak efficiency
    ax4 = axes[1, 1]
    df['Efficiency'] = 100 * df['GIOPS'] / adjusted_peak
    for tile in sorted(df['Tile'].unique()):
        if tile in colors:
            subset = df[df['Tile'] == tile].sort_values('N')
            ax4.plot(subset['N'], subset['Efficiency'], marker=markers.get(tile, 'o'),
                     color=colors.get(tile, 'gray'), linewidth=2, markersize=6,
                     label=f'Tile {tile}x{tile}x16')
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% Peak')
    ax4.axhline(y=75, color='orange', linestyle=':', alpha=0.5, label='75% Peak')
    ax4.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50% Peak')
    ax4.set_xlabel('Matrix Size (N×N)', fontsize=11)
    ax4.set_ylabel('Efficiency (% of Peak)', fontsize=11)
    ax4.set_title('Computational Efficiency vs Matrix Size', fontsize=12)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    ax4.set_ylim(0, min(100, df['Efficiency'].max() * 1.2))

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {output_png}")

    # Print summary table
    print("\n=== Performance Summary ===")
    print(f"Platform: {platform}")
    print(f"Config: {cores} cores @ {freq}GHz, {arch}")
    print(f"Cache: {l3_cache}")
    print(f"\nTheoretical Peak: {adjusted_peak} GIOP/s")
    print(f"Maximum Observed: {df['GIOPS'].max():.1f} GIOP/s ({100*df['GIOPS'].max()/adjusted_peak:.1f}%)")
    print("\nBest configuration per size:")
    print(best_per_size[['N', 'Tile', 'GIOPS']].to_string(index=False))

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot VNNI sweep results')
    parser.add_argument('--mnk', action='store_true',
                        help='Plot M,N,K sweep results (3-way heatmaps)')
    parser.add_argument('-i', '--input', default=None,
                        help='Input CSV file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output PNG file')
    args = parser.parse_args()

    if args.mnk:
        input_csv = args.input if args.input else '/tmp/mnk_sweep_results.csv'
        output_png = args.output if args.output else '/tmp/mnk_sweep_plot.png'
        plot_mnk_sweep(input_csv, output_png)
    else:
        input_csv = args.input if args.input else '/tmp/sweep_results.csv'
        output_png = args.output if args.output else '/tmp/vnni_sweep_plot.png'
        plot_sweep(input_csv, output_png)
