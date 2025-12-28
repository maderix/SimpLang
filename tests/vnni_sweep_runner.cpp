#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <chrono>

#define MEMREF_I8_2D(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)
#define MEMREF_I32_2D(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)

typedef int32_t (*MatmulFn)(
    int8_t*, int8_t*, int64_t, int64_t, int64_t,
    int8_t*, int8_t*, int64_t, int64_t, int64_t,
    int32_t*, int32_t*, int64_t, int64_t, int64_t
);

struct Bench { const char* name; const char* sym; int N; int iters; };

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: %s <vnni_sweep_all.so>\n", argv[0]); return 1; }
    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }

    Bench benches[] = {
        {"512_t64", "m512_t64", 512, 20},
        {"768_t64", "m768_t64", 768, 10},
        {"1024_t64", "m1024_t64", 1024, 5},
        {"1536_t64", "m1536_t64", 1536, 3},
        {"2048_t64", "m2048_t64", 2048, 2},
        {"3072_t64", "m3072_t64", 3072, 1},
        {"4096_t64", "m4096_t64", 4096, 1},
    };

    printf("=== Parallel INT8 VNNI Sweep (tile64) ===\n\n");
    printf("%-12s %6s %10s %10s\n", "Benchmark", "N", "Time(ms)", "GIOP/s");
    printf("--------------------------------------------\n");

    for (const auto& b : benches) {
        auto fn = (MatmulFn)dlsym(h, b.sym);
        if (!fn) { printf("%-12s not found\n", b.name); continue; }

        int8_t* A = (int8_t*)aligned_alloc(64, b.N * b.N);
        int8_t* B = (int8_t*)aligned_alloc(64, b.N * b.N);
        int32_t* C = (int32_t*)aligned_alloc(64, b.N * b.N * 4);
        memset(A, 1, b.N * b.N);
        memset(B, 1, b.N * b.N);

        // Warmup
        fn(MEMREF_I8_2D(A, b.N, b.N), MEMREF_I8_2D(B, b.N, b.N), MEMREF_I32_2D(C, b.N, b.N));

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < b.iters; i++) {
            fn(MEMREF_I8_2D(A, b.N, b.N), MEMREF_I8_2D(B, b.N, b.N), MEMREF_I32_2D(C, b.N, b.N));
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / b.iters;

        double ops = 2.0 * b.N * b.N * b.N;
        double giops = ops / ms / 1e6;
        printf("%-12s %6d %10.2f %10.0f\n", b.name, b.N, ms, giops);

        free(A); free(B); free(C);
    }
    printf("--------------------------------------------\n");
    dlclose(h);
    return 0;
}
