#!/usr/bin/env python3
"""
Run VNNI INT8 matmul sweep benchmark.
Compiles kernels, runs benchmarks with throttle delay, outputs CSV results.
Supports both square tile sweep and full M,N,K sweep modes.
"""

import argparse
import subprocess
import time
import os
import sys

# Default configuration
DEFAULT_SIZES = [512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384]
DEFAULT_TILES = [32, 64, 128, 256]
DEFAULT_THREADS = 8
DEFAULT_THROTTLE = 2.0  # seconds between sizes

# M,N,K sweep defaults
DEFAULT_M_TILES = [32, 64, 128, 256]
DEFAULT_N_TILES = [32, 64, 128, 256]
DEFAULT_K_TILES = [8, 16, 32, 64]
DEFAULT_MNK_SIZES = [1536, 5120, 7168]  # Best performing sizes

def run_mnk_sweep(sizes, m_tiles, n_tiles, k_tiles, threads=8, throttle_delay=2.0, output_csv='/tmp/mnk_sweep_results.csv'):
    """Run sweep with independent M, N, K tile sizes."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    simplang = os.path.join(project_root, 'build_mlir/src/simplang')

    if not os.path.exists(simplang):
        print(f"ERROR: simplang not found at {simplang}")
        sys.exit(1)

    # Generate sweep kernels
    configs = []
    for N in sizes:
        for M in m_tiles:
            for Nt in n_tiles:
                for K in k_tiles:
                    if M > N or Nt > N or K > N:
                        continue
                    configs.append((N, M, Nt, K))

    print(f"Generating {len(configs)} kernel configurations...")
    sl_file = '/tmp/mnk_sweep_all.sl'
    with open(sl_file, 'w') as f:
        for (N, M, Nt, K) in configs:
            fname = f"m{N}_tm{M}_tn{Nt}_tk{K}"
            f.write(f'''fn {fname}(i8[] A, i8[] B, i32[] C) -> i32 {{
    i8<{N}, {N}> At = tensor_from_array(A, 0i);
    i8<{N}, {N}> Bt = tensor_from_array(B, 0i);
    i32<{N}, {N}> Ct = tensor_from_array(C, 0i);
    @parallel @tile({M}, {Nt}, {K}) @lower("vnni.i8_matmul")
    tensor_matmul_out(At, Bt, Ct);
    return Ct[0i, 0i];
}}
''')

    # Compile
    print("Compiling kernels...")
    obj_file = '/tmp/mnk_sweep_all.o'
    so_file = '/tmp/mnk_sweep_all.so'

    result = subprocess.run(
        [simplang, sl_file, '--emit-mlir', '--llvm-vectorize', '-o', obj_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        sys.exit(1)

    result = subprocess.run(
        ['gcc', '-shared', '-o', so_file, obj_file, '-lm', '-liomp5', '-L/usr/lib/llvm-14/lib'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Linking failed:\n{result.stderr}")
        sys.exit(1)

    # Generate and compile runner
    print("Generating benchmark runner...")
    runner_cpp = '/tmp/mnk_sweep_runner.cpp'
    runner_bin = '/tmp/mnk_sweep_runner'

    # Build config arrays for C++
    config_strs = ', '.join(f'{{{n}, {m}, {nt}, {k}}}' for (n, m, nt, k) in configs)

    with open(runner_cpp, 'w') as f:
        f.write(f'''
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dlfcn.h>
#include <chrono>
#include <unistd.h>
#include <map>
#include <string>

#define MEMREF_I8(ptr, sz) (ptr), (ptr), 0, (int64_t)(sz), 1
#define MEMREF_I32(ptr, sz) (ptr), (ptr), 0, (int64_t)(sz), 1

typedef int32_t (*Fn)(int8_t*, int8_t*, int64_t, int64_t, int64_t,
                      int8_t*, int8_t*, int64_t, int64_t, int64_t,
                      int32_t*, int32_t*, int64_t, int64_t, int64_t);

struct Config {{ int N, M, Nt, K; }};

int main(int argc, char** argv) {{
    float throttle = {throttle_delay}f;
    if (argc > 1) throttle = atof(argv[1]);

    void* h = dlopen("{so_file}", RTLD_NOW);
    if (!h) {{ printf("dlopen: %s\\n", dlerror()); return 1; }}

    Config configs[] = {{{config_strs}}};
    int nconfigs = sizeof(configs)/sizeof(configs[0]);
    double peak = 2560.0;

    printf("N,M_tile,N_tile,K_tile,Time_ms,GIOPS,Percent_Peak\\n");
    fflush(stdout);

    std::map<int, std::pair<int8_t*, int8_t*>> ab_cache;
    std::map<int, int32_t*> c_cache;
    int last_N = 0;

    for (int ci = 0; ci < nconfigs; ci++) {{
        int N = configs[ci].N;
        int M = configs[ci].M;
        int Nt = configs[ci].Nt;
        int K = configs[ci].K;
        size_t sz = (size_t)N * N;

        // Allocate/reuse buffers
        int8_t *A, *B;
        int32_t *C;
        if (ab_cache.find(N) == ab_cache.end()) {{
            A = (int8_t*)aligned_alloc(64, sz);
            B = (int8_t*)aligned_alloc(64, sz);
            C = (int32_t*)aligned_alloc(64, sz * 4);
            if (!A || !B || !C) {{ fprintf(stderr, "OOM for N=%d\\n", N); continue; }}
            for (size_t i = 0; i < sz; i++) {{ A[i] = 1; B[i] = 1; }}
            ab_cache[N] = {{A, B}};
            c_cache[N] = C;
        }} else {{
            A = ab_cache[N].first;
            B = ab_cache[N].second;
            C = c_cache[N];
        }}

        char fname[128];
        sprintf(fname, "m%d_tm%d_tn%d_tk%d", N, M, Nt, K);
        Fn fn = (Fn)dlsym(h, fname);
        if (!fn) {{ fprintf(stderr, "Symbol not found: %s\\n", fname); continue; }}

        // Warmup
        fn(MEMREF_I8(A, sz), MEMREF_I8(B, sz), MEMREF_I32(C, sz));

        int iters = N <= 2048 ? 5 : 3;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {{
            fn(MEMREF_I8(A, sz), MEMREF_I8(B, sz), MEMREF_I32(C, sz));
        }}
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        double gops = 2.0 * N * N * N / 1e9;
        double giops = gops / (ms / 1000.0);
        printf("%d,%d,%d,%d,%g,%g,%.4f\\n", N, M, Nt, K, ms, giops, 100*giops/peak);
        fflush(stdout);

        // Throttle between different matrix sizes
        if (ci < nconfigs - 1 && configs[ci+1].N != N && throttle > 0) {{
            usleep((int)(throttle * 1000000));
        }}
    }}

    // Cleanup
    for (auto& p : ab_cache) {{
        free(p.second.first);
        free(p.second.second);
    }}
    for (auto& p : c_cache) {{
        free(p.second);
    }}
    return 0;
}}
''')

    result = subprocess.run(
        ['g++', '-O2', '-o', runner_bin, runner_cpp, '-ldl'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Runner compilation failed:\n{result.stderr}")
        sys.exit(1)

    # Run benchmark
    print(f"Running M,N,K sweep with {threads} threads, {throttle_delay}s throttle delay...")
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    env['LD_LIBRARY_PATH'] = '/usr/lib/llvm-14/lib:' + env.get('LD_LIBRARY_PATH', '')

    with open(output_csv, 'w') as out:
        result = subprocess.run(
            [runner_bin, str(throttle_delay)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out.write(result.stdout)
        print(result.stdout)

    print(f"\nResults saved to {output_csv}")
    return output_csv


def run_sweep(sizes, tiles, threads=8, throttle_delay=2.0, output_csv='/tmp/sweep_results.csv'):
    """Run sweep with square tiles (original mode)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    simplang = os.path.join(project_root, 'build_mlir/src/simplang')

    if not os.path.exists(simplang):
        print(f"ERROR: simplang not found at {simplang}")
        sys.exit(1)

    # Generate sweep kernels
    print(f"Generating kernels for {len(sizes)} sizes x {len(tiles)} tiles...")
    sl_file = '/tmp/sweep_all.sl'
    with open(sl_file, 'w') as f:
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

    # Compile
    print("Compiling kernels...")
    obj_file = '/tmp/sweep_all.o'
    so_file = '/tmp/sweep_all.so'
    
    result = subprocess.run(
        [simplang, sl_file, '--emit-mlir', '--llvm-vectorize', '-o', obj_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        sys.exit(1)

    result = subprocess.run(
        ['gcc', '-shared', '-o', so_file, obj_file, '-lm', '-liomp5', '-L/usr/lib/llvm-14/lib'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Linking failed:\n{result.stderr}")
        sys.exit(1)

    # Generate and compile runner
    print("Generating benchmark runner...")
    runner_cpp = '/tmp/sweep_runner.cpp'
    runner_bin = '/tmp/sweep_runner'
    
    sizes_str = ', '.join(str(s) for s in sizes)
    tiles_str = ', '.join(str(t) for t in tiles)
    
    with open(runner_cpp, 'w') as f:
        f.write(f'''
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dlfcn.h>
#include <chrono>
#include <unistd.h>

#define MEMREF_I8(ptr, sz) (ptr), (ptr), 0, (int64_t)(sz), 1
#define MEMREF_I32(ptr, sz) (ptr), (ptr), 0, (int64_t)(sz), 1

typedef int32_t (*Fn)(int8_t*, int8_t*, int64_t, int64_t, int64_t,
                      int8_t*, int8_t*, int64_t, int64_t, int64_t,
                      int32_t*, int32_t*, int64_t, int64_t, int64_t);

int main(int argc, char** argv) {{
    float throttle = {throttle_delay}f;
    if (argc > 1) throttle = atof(argv[1]);

    void* h = dlopen("{so_file}", RTLD_NOW);
    if (!h) {{ printf("dlopen: %s\\n", dlerror()); return 1; }}

    int sizes[] = {{{sizes_str}}};
    int tiles[] = {{{tiles_str}}};
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);
    int ntiles = sizeof(tiles)/sizeof(tiles[0]);
    double peak = 2560.0;

    printf("N,Tile,Time_ms,GIOPS,Percent_Peak\\n");
    fflush(stdout);

    for (int si = 0; si < nsizes; si++) {{
        int N = sizes[si];
        size_t sz = (size_t)N * N;

        int8_t* A = (int8_t*)aligned_alloc(64, sz);
        int8_t* B = (int8_t*)aligned_alloc(64, sz);
        int32_t* C = (int32_t*)aligned_alloc(64, sz * 4);
        if (!A || !B || !C) {{ fprintf(stderr, "OOM for N=%d\\n", N); continue; }}
        for (size_t i = 0; i < sz; i++) {{ A[i] = 1; B[i] = 1; }}

        for (int ti = 0; ti < ntiles; ti++) {{
            int T = tiles[ti];
            if (T > N) continue;

            char fname[64];
            sprintf(fname, "m%d_t%d", N, T);
            Fn fn = (Fn)dlsym(h, fname);
            if (!fn) continue;

            // Warmup
            fn(MEMREF_I8(A, sz), MEMREF_I8(B, sz), MEMREF_I32(C, sz));

            int iters = N <= 1024 ? 10 : (N <= 4096 ? 5 : 3);
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {{
                fn(MEMREF_I8(A, sz), MEMREF_I8(B, sz), MEMREF_I32(C, sz));
            }}
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
            double gops = 2.0 * N * N * N / 1e9;
            double giops = gops / (ms / 1000.0);
            printf("%d,%d,%g,%g,%.4f\\n", N, T, ms, giops, 100*giops/peak);
            fflush(stdout);
        }}

        free(A); free(B); free(C);

        // Throttle delay between sizes to reduce thermal throttling
        if (si < nsizes - 1 && throttle > 0) {{
            usleep((int)(throttle * 1000000));
        }}
    }}
    return 0;
}}
''')

    result = subprocess.run(
        ['g++', '-O2', '-o', runner_bin, runner_cpp, '-ldl'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Runner compilation failed:\n{result.stderr}")
        sys.exit(1)

    # Run benchmark
    print(f"Running sweep with {threads} threads, {throttle_delay}s throttle delay...")
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    env['LD_LIBRARY_PATH'] = '/usr/lib/llvm-14/lib:' + env.get('LD_LIBRARY_PATH', '')

    with open(output_csv, 'w') as out:
        result = subprocess.run(
            [runner_bin, str(throttle_delay)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out.write(result.stdout)
        print(result.stdout)

    print(f"\nResults saved to {output_csv}")
    return output_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VNNI sweep benchmark')
    parser.add_argument('--mnk', action='store_true',
                        help='Run M,N,K independent tile sweep (default: square tiles)')
    parser.add_argument('--sizes', type=int, nargs='+', default=None,
                        help='Matrix sizes to sweep')
    parser.add_argument('--tiles', type=int, nargs='+', default=DEFAULT_TILES,
                        help='Tile sizes to sweep (square mode)')
    parser.add_argument('--m-tiles', type=int, nargs='+', default=DEFAULT_M_TILES,
                        help='M tile sizes (MNK mode)')
    parser.add_argument('--n-tiles', type=int, nargs='+', default=DEFAULT_N_TILES,
                        help='N tile sizes (MNK mode)')
    parser.add_argument('--k-tiles', type=int, nargs='+', default=DEFAULT_K_TILES,
                        help='K tile sizes (MNK mode)')
    parser.add_argument('-t', '--threads', type=int, default=DEFAULT_THREADS,
                        help='Number of OMP threads')
    parser.add_argument('--throttle', type=float, default=DEFAULT_THROTTLE,
                        help='Delay in seconds between matrix sizes (reduces throttling)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output CSV file')
    args = parser.parse_args()

    if args.mnk:
        sizes = args.sizes if args.sizes else DEFAULT_MNK_SIZES
        output = args.output if args.output else '/tmp/mnk_sweep_results.csv'
        run_mnk_sweep(sizes, args.m_tiles, args.n_tiles, args.k_tiles,
                      args.threads, args.throttle, output)
    else:
        sizes = args.sizes if args.sizes else DEFAULT_SIZES
        output = args.output if args.output else '/tmp/sweep_results.csv'
        run_sweep(sizes, args.tiles, args.threads, args.throttle, output)
