#!/usr/bin/env python3
"""
Analyze VNNI sweep results and provide optimization insights.
"""

import argparse
import pandas as pd

def analyze_sweep(input_csv='/tmp/sweep_results.csv'):
    df = pd.read_csv(input_csv)
    peak = 2560

    print("=== VNNI INT8 MatMul Analysis ===\n")

    print("Top 10 configurations:")
    top10 = df.nlargest(10, 'GIOPS')[['N', 'Tile', 'GIOPS']]
    for _, r in top10.iterrows():
        print(f"  N={r['N']:5}, Tile={r['Tile']:3}: {r['GIOPS']:7.0f} GIOP/s ({100*r['GIOPS']/peak:5.1f}%)")

    max_giops = df['GIOPS'].max()
    print(f"\nPeak observed: {max_giops:.0f} GIOP/s ({100*max_giops/peak:.1f}% of theoretical)")
    print(f"Gap to 2000 GIOP/s: {2000 - max_giops:.0f} GIOP/s")
    print(f"Gap to peak (2560): {peak - max_giops:.0f} GIOP/s")

    # Memory bandwidth analysis for large sizes
    print("\n=== Memory Bandwidth Analysis ===")
    large_sizes = [n for n in df['N'].unique() if n >= 4096]
    for N in sorted(large_sizes)[:5]:
        rows = df[(df['N']==N) & (df['Tile']==64)]
        if len(rows) > 0:
            row = rows.iloc[0]
            data_size_mb = (N*N*1 + N*N*1 + N*N*4) / 1e6  # A + B + C
            time_s = (2.0 * N**3 / 1e9) / row['GIOPS']
            bw = data_size_mb / time_s / 1000  # GB/s
            print(f"  N={N:5}: {data_size_mb:6.0f}MB data, {bw:5.1f} GB/s effective BW, {row['GIOPS']:.0f} GIOP/s")

    # Best tile per size
    print("\n=== Best Tile per Size ===")
    best = df.loc[df.groupby('N')['GIOPS'].idxmax()][['N', 'Tile', 'GIOPS']]
    for _, r in best.iterrows():
        print(f"  N={r['N']:5}: Tile {r['Tile']:3}x{r['Tile']:3} -> {r['GIOPS']:7.0f} GIOP/s")

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze VNNI sweep results')
    parser.add_argument('-i', '--input', default='/tmp/sweep_results.csv',
                        help='Input CSV file')
    args = parser.parse_args()

    analyze_sweep(args.input)
