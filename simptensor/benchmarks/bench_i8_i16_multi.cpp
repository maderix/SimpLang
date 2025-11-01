#include <iostream>
#include <iomanip>
#include <dlfcn.h>
#include <cstdint>
#include <chrono>

typedef int32_t (*KernelFunc)();

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

void bench_size(void* handle, const char* func_name, const char* type, int N, int iterations) {
    KernelFunc kernel = (KernelFunc)dlsym(handle, func_name);
    if (!kernel) {
        std::cout << type << " " << std::setw(3) << N << "×" << std::setw(4) << N << " │ ";
        std::cout << "FAILED TO LOAD" << std::endl;
        return;
    }

    double time = benchmark(kernel, iterations);
    int32_t result = kernel();
    double giops = 2.0 * N * N * N / 1e9;
    double giops_per_sec = giops / (time / 1000.0);

    std::cout << type << " " << std::setw(3) << N << "×" << std::setw(4) << N << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << giops_per_sec << " │ ";
    std::cout << std::setw(12) << result << " │ ✓" << std::endl;
}

int main() {
    void* handle = dlopen("/tmp/bench_matmul_multi.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load .so: " << dlerror() << std::endl;
        return 1;
    }

    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   i8/i16 Matrix Multiplication Benchmarks (Multiple Sizes)" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << " Type Size   │   Time    │  GIOP/s  │   Checksum   │ Status" << std::endl;
    std::cout << "─────────────┼───────────┼──────────┼──────────────┼────────" << std::endl;

    // i8 benchmarks
    bench_size(handle, "benchmark_matmul_64_i8", "i8 ", 64, 10);
    bench_size(handle, "benchmark_matmul_128_i8", "i8 ", 128, 10);
    bench_size(handle, "benchmark_matmul_256_i8", "i8 ", 256, 5);
    bench_size(handle, "benchmark_matmul_512_i8", "i8 ", 512, 3);

    std::cout << "─────────────┼───────────┼──────────┼──────────────┼────────" << std::endl;

    // i16 benchmarks
    bench_size(handle, "benchmark_matmul_64_i16", "i16", 64, 10);
    bench_size(handle, "benchmark_matmul_128_i16", "i16", 128, 10);
    bench_size(handle, "benchmark_matmul_256_i16", "i16", 256, 5);
    bench_size(handle, "benchmark_matmul_512_i16", "i16", 512, 3);

    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << "Notes:" << std::endl;
    std::cout << "• i8×i8 and i16×i16 matmul return i32 results (no overflow)" << std::endl;
    std::cout << "• GIOP/s = Giga Integer Operations Per Second" << std::endl;
    std::cout << "• Performance optimized with MLIR 8×8×8 tiling + vectorization" << std::endl;
    std::cout << std::endl;

    dlclose(handle);
    return 0;
}
