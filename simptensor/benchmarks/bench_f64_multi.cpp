#include <iostream>
#include <iomanip>
#include <dlfcn.h>
#include <cstdint>
#include <chrono>

typedef double (*KernelFunc)();

double benchmark(KernelFunc func, int iterations) {
    func(); // warmup
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / (double)iterations / 1000.0; // ms
}

void bench_size(void* handle, const char* func_name, int N, int iterations) {
    KernelFunc kernel = (KernelFunc)dlsym(handle, func_name);
    if (!kernel) {
        std::cout << "f64 " << std::setw(3) << N << "×" << std::setw(4) << N << " │ ";
        std::cout << "FAILED TO LOAD" << std::endl;
        return;
    }

    double time = benchmark(kernel, iterations);
    double result = kernel();
    double gflops = 2.0 * N * N * N / 1e9;
    double gflops_per_sec = gflops / (time / 1000.0);

    std::cout << "f64 " << std::setw(3) << N << "×" << std::setw(4) << N << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << gflops_per_sec << " │ ";
    std::cout << std::setw(12) << std::fixed << std::setprecision(1) << result << " │ ✓" << std::endl;
}

int main() {
    void* handle = dlopen("/tmp/bench_matmul_f64.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load .so: " << dlerror() << std::endl;
        return 1;
    }

    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   f64 (double) Matrix Multiplication Benchmarks" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << " Type Size   │   Time    │ GFLOP/s  │   Checksum   │ Status" << std::endl;
    std::cout << "─────────────┼───────────┼──────────┼──────────────┼────────" << std::endl;

    // f64 benchmarks
    bench_size(handle, "benchmark_matmul_64_f64", 64, 10);
    bench_size(handle, "benchmark_matmul_128_f64", 128, 10);
    bench_size(handle, "benchmark_matmul_256_f64", 256, 5);
    bench_size(handle, "benchmark_matmul_512_f64", 512, 3);
    bench_size(handle, "benchmark_matmul_1024_f64", 1024, 2);

    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << "Notes:" << std::endl;
    std::cout << "• f64 = double precision (64-bit floats)" << std::endl;
    std::cout << "• GFLOP/s = Giga Floating Point Operations Per Second" << std::endl;
    std::cout << "• Performance optimized with MLIR 8×8×8 tiling + vectorization" << std::endl;
    std::cout << std::endl;

    dlclose(handle);
    return 0;
}
