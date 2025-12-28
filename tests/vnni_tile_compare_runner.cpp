#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <dlfcn.h>
#include <chrono>

#define MEMREF_I8_2D(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)
#define MEMREF_I32_2D(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)

typedef int32_t (*MatmulFn)(
    int8_t*, int8_t*, int64_t, int64_t, int64_t,
    int8_t*, int8_t*, int64_t, int64_t, int64_t,
    int32_t*, int32_t*, int64_t, int64_t, int64_t
);

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: %s <vnni_sweep_all.so>\n", argv[0]); return 1; }
    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }

    int sizes[] = {1024, 2048, 4096};
    int tiles[] = {32, 64, 128, 256};
    int iters[] = {5, 2, 1};

    printf("=== Tile Size Comparison (Parallel VNNI) ===\n\n");
    printf("%-6s %8s %8s %8s %8s\n", "Size", "t32", "t64", "t128", "t256");
    printf("----------------------------------------------\n");

    for (int si = 0; si < 3; si++) {
        int N = sizes[si];
        int it = iters[si];

        int8_t* A = (int8_t*)aligned_alloc(64, (size_t)N * N);
        int8_t* B = (int8_t*)aligned_alloc(64, (size_t)N * N);
        int32_t* C = (int32_t*)aligned_alloc(64, (size_t)N * N * 4);
        memset(A, 1, (size_t)N * N);
        memset(B, 1, (size_t)N * N);

        printf("%-6d", N);

        for (int t : tiles) {
            char sym[32];
            snprintf(sym, sizeof(sym), "m%d_t%d", N, t);
            auto fn = (MatmulFn)dlsym(h, sym);
            if (!fn) { printf(" %8s", "N/A"); continue; }

            fn(MEMREF_I8_2D(A, N, N), MEMREF_I8_2D(B, N, N), MEMREF_I32_2D(C, N, N));

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < it; i++) {
                fn(MEMREF_I8_2D(A, N, N), MEMREF_I8_2D(B, N, N), MEMREF_I32_2D(C, N, N));
            }
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count() / it;
            double giops = (2.0 * N * N * N) / ms / 1e6;
            printf(" %8.0f", giops);
        }
        printf("\n");
        free(A); free(B); free(C);
    }
    printf("----------------------------------------------\n");
    printf("(GIOP/s - higher is better)\n");
    dlclose(h);
    return 0;
}
