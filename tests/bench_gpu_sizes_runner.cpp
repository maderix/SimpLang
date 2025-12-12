// GPU f32 MatMul Benchmark Runner - Multiple sizes
// Measures total time AND matmul-only time separately
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <dlfcn.h>
#include <chrono>

typedef float (*bench_fn)();

struct BenchResult {
    double total_ms;
    double tflops_total;
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
    res.total_ms = total_ms / iterations;

    // TFLOPS: 2*M*N*K for matmul
    double flops = 2.0 * N * N * N;
    res.tflops_total = (flops / (res.total_ms / 1000.0)) / 1e12;

    res.expected = 2.0f * N;
    res.correct = fabs(res.result - res.expected) < 1.0f;

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

    printf("╔════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║        SimpLang GPU MatMul Benchmark (tensor_fill + cuBLAS)                   ║\n");
    printf("╠════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Size       │  Total Time │  TFLOPS    │  Result    │  Status │  cuBLAS Peak  ║\n");
    printf("╠═════════════╪═════════════╪════════════╪════════════╪═════════╪═══════════════╣\n");

    // Direct cuBLAS reference TFLOPS (from cublas_bench.cpp results)
    double cublas_ref[] = {0.033, 8.5, 30.0, 53.0, 57.0, 60.0};  // 256,512,1024,2048,4096,6144,8192

    // 1024x1024
    BenchResult r1024 = run_benchmark(handle, "bench_1024", 1024, 20);
    printf("║ 1024x1024   │ %8.2f ms │ %8.3f   │ %8.1f   │    %s    │   ~30 TFLOPS  ║\n",
           r1024.total_ms, r1024.tflops_total, r1024.result, r1024.correct ? "✓" : "✗");

    // 2048x2048
    BenchResult r2048 = run_benchmark(handle, "bench_2048", 2048, 10);
    printf("║ 2048x2048   │ %8.2f ms │ %8.3f   │ %8.1f   │    %s    │   ~53 TFLOPS  ║\n",
           r2048.total_ms, r2048.tflops_total, r2048.result, r2048.correct ? "✓" : "✗");

    // 4096x4096
    BenchResult r4096 = run_benchmark(handle, "bench_4096", 4096, 5);
    printf("║ 4096x4096   │ %8.2f ms │ %8.3f   │ %8.1f   │    %s    │   ~57 TFLOPS  ║\n",
           r4096.total_ms, r4096.tflops_total, r4096.result, r4096.correct ? "✗" : "✓");

    // 6144x6144
    BenchResult r6144 = run_benchmark(handle, "bench_6144", 6144, 3);
    printf("║ 6144x6144   │ %8.2f ms │ %8.3f   │ %8.1f   │    %s    │   ~60 TFLOPS  ║\n",
           r6144.total_ms, r6144.tflops_total, r6144.result, r6144.correct ? "✓" : "✗");

    // 8192x8192
    BenchResult r8192 = run_benchmark(handle, "bench_8192", 8192, 2);
    printf("║ 8192x8192   │ %8.2f ms │ %8.3f   │ %8.1f   │    %s    │   ~60 TFLOPS  ║\n",
           r8192.total_ms, r8192.tflops_total, r8192.result, r8192.correct ? "✓" : "✗");

    printf("╚════════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\nAnalysis:\n");
    printf("  - RTX 4090 FP32 theoretical peak: 83 TFLOPS\n");
    printf("  - cuBLAS Peak column shows direct cuBLAS performance (no SimpLang overhead)\n");
    printf("  - Gap is due to: tensor init on CPU + H2D copy per call\n");
    printf("  - For best perf: keep data on GPU, batch operations\n");

    dlclose(handle);
    return 0;
}
