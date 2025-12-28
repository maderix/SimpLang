// Benchmark runner: Lum vs SimpLang Internal Tiling
// Compile and run with:
//   g++ -O3 -o bench_runner bench_lum_vs_internal_runner.cpp -ldl
//   ./bench_runner <kernel.so> [iterations]

#include <dlfcn.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

typedef float (*KernelFunc)();

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so> [iterations=10]" << std::endl;
        return 1;
    }

    const char* soPath = argv[1];
    int iterations = (argc > 2) ? std::atoi(argv[2]) : 10;

    // Load kernel
    void* handle = dlopen(soPath, RTLD_NOW);
    if (!handle) {
        std::cerr << "Error loading " << soPath << ": " << dlerror() << std::endl;
        return 1;
    }

    KernelFunc kernel = (KernelFunc)dlsym(handle, "matmul_512");
    if (!kernel) {
        std::cerr << "Error finding matmul_512: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    std::cout << "Loaded: " << soPath << std::endl;
    std::cout << "Running " << iterations << " iterations..." << std::endl;

    // Warmup
    float result = kernel();
    std::cout << "Warmup result (checksum): " << result << std::endl;

    // Benchmark
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        result = kernel();
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    // Statistics
    std::sort(times.begin(), times.end());
    double median = times[iterations / 2];
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / iterations;
    double min_time = times.front();
    double max_time = times.back();

    // GFLOPS calculation: 2*M*N*K for matmul
    double flops = 2.0 * 512 * 512 * 512;
    double gflops_median = (flops / (median / 1000.0)) / 1e9;
    double gflops_best = (flops / (min_time / 1000.0)) / 1e9;

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Matrix size: 512x512" << std::endl;
    std::cout << "Checksum: " << result << std::endl;
    std::cout << "\nTiming (ms):" << std::endl;
    std::cout << "  Min:    " << min_time << std::endl;
    std::cout << "  Median: " << median << std::endl;
    std::cout << "  Mean:   " << mean << std::endl;
    std::cout << "  Max:    " << max_time << std::endl;
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Best:   " << gflops_best << " GFLOPS" << std::endl;
    std::cout << "  Median: " << gflops_median << " GFLOPS" << std::endl;

    dlclose(handle);
    return 0;
}
