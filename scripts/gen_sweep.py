#!/usr/bin/env python3
"""
Generate SimpLang VNNI INT8 matmul sweep kernels.
Uses tensor_matmul_out for zero-copy output.
"""

import argparse

# Default sweep configuration
DEFAULT_SIZES = [512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384]
DEFAULT_TILES = [32, 64, 128, 256]

def generate_sweep(sizes, tiles, output_file='/tmp/sweep_all.sl'):
    count = 0
    with open(output_file, 'w') as f:
        for N in sizes:
            for T in tiles:
                if T > N:
                    continue
                fname = f"m{N}_t{T}"
                f.write(f'''fn {fname}(i8[] A, i8[] B, i32[] C) -> i32 {{
    i8<{N}, {N}> At = tensor_from_array(A, 0i);
    i8<{N}, {N}> Bt = tensor_from_array(B, 0i);
    i32<{N}, {N}> Ct = tensor_from_array(C, 0i);
    @parallel @tile({T}, {T}, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}}
''')
                count += 1
    print(f"Generated {count} functions to {output_file}")
    return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate VNNI sweep kernels')
    parser.add_argument('--sizes', type=int, nargs='+', default=DEFAULT_SIZES,
                        help='Matrix sizes to sweep')
    parser.add_argument('--tiles', type=int, nargs='+', default=DEFAULT_TILES,
                        help='Tile sizes to sweep')
    parser.add_argument('-o', '--output', default='/tmp/sweep_all.sl',
                        help='Output .sl file')
    args = parser.parse_args()

    generate_sweep(args.sizes, args.tiles, args.output)
