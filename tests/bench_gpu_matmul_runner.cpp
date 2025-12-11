#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <dlfcn.h>
#include <chrono>

typedef float (*kernel_main_fn)();

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel.so> [iterations]\n", argv[0]);
        return 1;
    }

    int iterations = (argc >= 3) ? atoi(argv[2]) : 10;

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "Error loading kernel: %s\n", dlerror());
        return 1;
    }

    kernel_main_fn kernel_main = (kernel_main_fn)dlsym(handle, "kernel_main");
    if (!kernel_main) {
        fprintf(stderr, "Error finding kernel_main: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }

    printf("=== GPU MatMul Benchmark ===\n");
    printf("Matrix size: 1024x1024\n");
    printf("Iterations: %d\n\n", iterations);

    // Warmup
    printf("Warming up...\n");
    float result = kernel_main();
    printf("Warmup result: %.1f (expected: 2048.0)\n\n", result);

    // Benchmark
    printf("Running benchmark...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        result = kernel_main();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double total_ms = duration.count() / 1000.0;
    double avg_ms = total_ms / iterations;
    
    // GFLOPS calculation: 2*M*N*K for matmul
    double flops = 2.0 * 1024 * 1024 * 1024;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;
    
    printf("\n=== Results ===\n");
    printf("Total time: %.2f ms\n", total_ms);
    printf("Avg per iteration: %.2f ms\n", avg_ms);
    printf("Throughput: %.2f GFLOPS\n", gflops);
    printf("Final result: %.1f\n", result);

    dlclose(handle);
    return 0;
}
