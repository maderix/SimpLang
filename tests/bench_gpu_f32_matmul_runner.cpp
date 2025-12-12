// GPU f32 MatMul Benchmark Runner
// Compares cuBLAS performance across different matrix sizes

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <dlfcn.h>
#include <chrono>

typedef float (*bench_fn)();

struct BenchResult {
    double avg_ms;
    double gflops;
    float result;
    float expected;
    bool correct;
};

BenchResult run_benchmark(void* handle, const char* func_name, int N, int iterations) {
    BenchResult res = {0, 0, 0, 0, false};

    bench_fn fn = (bench_fn)dlsym(handle, func_name);
    if (!fn) {
        fprintf(stderr, "Error finding %s: %s\n", func_name, dlerror());
        return res;
    }

    // Warmup
    res.result = fn();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        res.result = fn();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    res.avg_ms = total_ms / iterations;

    // GFLOPS: 2*M*N*K for matmul
    double flops = 2.0 * N * N * N;
    res.gflops = (flops / (res.avg_ms / 1000.0)) / 1e9;

    // Expected: N * 1.0 * 2.0 = 2*N
    res.expected = 2.0f * N;
    res.correct = fabs(res.result - res.expected) < 0.01f;

    return res;
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

    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║          GPU f32 MatMul Benchmark (cuBLAS on RTX 4090)               ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Size       │  Avg Time   │   TFLOPS   │   Result   │    Status      ║\n");
    printf("╠═════════════╪═════════════╪════════════╪════════════╪════════════════╣\n");

    // 256x256
    BenchResult r256 = run_benchmark(handle, "bench_256", 256, 100);
    printf("║  256x256    │ %8.3f ms │ %8.3f   │ %8.1f   │      %s        ║\n",
           r256.avg_ms, r256.gflops/1000.0, r256.result, r256.correct ? "✓" : "✗");

    // 512x512
    BenchResult r512 = run_benchmark(handle, "bench_512", 512, 50);
    printf("║  512x512    │ %8.3f ms │ %8.3f   │ %8.1f   │      %s        ║\n",
           r512.avg_ms, r512.gflops/1000.0, r512.result, r512.correct ? "✓" : "✗");

    // 1024x1024
    BenchResult r1024 = run_benchmark(handle, "bench_1024", 1024, 20);
    printf("║ 1024x1024   │ %8.3f ms │ %8.3f   │ %8.1f   │      %s        ║\n",
           r1024.avg_ms, r1024.gflops/1000.0, r1024.result, r1024.correct ? "✓" : "✗");

    // 2048x2048
    BenchResult r2048 = run_benchmark(handle, "bench_2048", 2048, 10);
    printf("║ 2048x2048   │ %8.3f ms │ %8.3f   │ %8.1f   │      %s        ║\n",
           r2048.avg_ms, r2048.gflops/1000.0, r2048.result, r2048.correct ? "✓" : "✗");

    // 4096x4096
    BenchResult r4096 = run_benchmark(handle, "bench_4096", 4096, 5);
    printf("║ 4096x4096   │ %8.3f ms │ %8.3f   │ %8.1f   │      %s        ║\n",
           r4096.avg_ms, r4096.gflops/1000.0, r4096.result, r4096.correct ? "✓" : "✗");

    // 6144x6144
    BenchResult r6144 = run_benchmark(handle, "bench_6144", 6144, 3);
    printf("║ 6144x6144   │ %8.3f ms │ %8.3f   │ %8.1f   │      %s        ║\n",
           r6144.avg_ms, r6144.gflops/1000.0, r6144.result, r6144.correct ? "✓" : "✗");

    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    printf("\nNotes:\n");
    printf("  - TFLOPS = 2*N³ / time (higher is better)\n");
    printf("  - RTX 4090 FP32 peak: ~83 TFLOPS\n");
    printf("  - cuBLAS efficiency improves with larger matrices\n");
    printf("  - Memory: 4096x4096 = 64MB per matrix, 6144x6144 = 144MB per matrix\n");

    dlclose(handle);
    return 0;
}
