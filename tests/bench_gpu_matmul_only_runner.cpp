// GPU MatMul-Only Benchmark Runner
// Host pre-fills arrays, measures pure matmul time
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <dlfcn.h>
#include <chrono>
#include <vector>

// MLIR memref descriptor: (allocPtr, alignedPtr, offset, size, stride)
#define MEMREF_F32_PARAMS float*, float*, int64_t, int64_t, int64_t
#define MEMREF_F32(ptr, size) (ptr), (ptr), 0, (int64_t)(size), 1

typedef float (*bench_fn)(MEMREF_F32_PARAMS, MEMREF_F32_PARAMS, MEMREF_F32_PARAMS);
typedef void (*sync_fn)();

void run_matmul_bench(void* handle, const char* func_name, int N, int iterations, sync_fn sync_output) {
    bench_fn fn = (bench_fn)dlsym(handle, func_name);
    if (!fn) {
        fprintf(stderr, "Error finding %s: %s\n", func_name, dlerror());
        return;
    }

    size_t size = (size_t)N * N;
    std::vector<float> A(size, 1.0f);
    std::vector<float> B(size, 2.0f);
    std::vector<float> C(size, 0.0f);

    // Warmup (with sync to ensure GPU is ready)
    float result = fn(MEMREF_F32(A.data(), size), MEMREF_F32(B.data(), size), MEMREF_F32(C.data(), size));
    sync_output();

    // Benchmark - no D2H during timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        result = fn(MEMREF_F32(A.data(), size), MEMREF_F32(B.data(), size), MEMREF_F32(C.data(), size));
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Sync final result for verification (not timed)
    sync_output();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;

    double flops = 2.0 * N * N * N;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    float expected = 2.0f * N;
    bool correct = fabs(result - expected) < 1.0f;

    printf("║ %5dx%-5d │ %8.3f ms │ %8.3f   │ %8.1f   │    %s    ║\n",
           N, N, avg_ms, tflops, result, correct ? "✓" : "✗");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel.so>\n", argv[0]);
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "Error loading kernel: %s\n", dlerror());
        return 1;
    }

    // Get sync function from runtime
    sync_fn sync_output = (sync_fn)dlsym(handle, "simp_gpu_sync_output");
    if (!sync_output) {
        fprintf(stderr, "Error: simp_gpu_sync_output not found\n");
        return 1;
    }

    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║      SimpLang GPU MatMul Benchmark (compute only, no D2H)            ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Size       │  Avg Time   │   TFLOPS   │   Result   │    Status      ║\n");
    printf("╠═════════════╪═════════════╪════════════╪════════════╪════════════════╣\n");

    run_matmul_bench(handle, "bench_matmul_2048", 2048, 10, sync_output);
    run_matmul_bench(handle, "bench_matmul_4096", 4096, 5, sync_output);
    run_matmul_bench(handle, "bench_matmul_8192", 8192, 3, sync_output);

    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    printf("\nReference (Direct cuBLAS on RTX 4090):\n");
    printf("  2048x2048: ~53 TFLOPS\n");
    printf("  4096x4096: ~57 TFLOPS\n");
    printf("  8192x8192: ~60 TFLOPS\n");

    dlclose(handle);
    return 0;
}
