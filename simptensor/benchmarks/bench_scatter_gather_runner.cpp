//===- bench_scatter_gather_runner.cpp - Scatter/Gather Benchmark Runner -===//
//
// Benchmark runner for scatter/gather operations
// Compares SimpLang performance vs native C++ baseline
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>

typedef float (*BenchFunc)();

struct Benchmark {
    const char* name;
    const char* func_name;
    const char* category;
};

double runBenchmark(void* handle, const Benchmark& bench, int iterations) {
    BenchFunc func = (BenchFunc)dlsym(handle, bench.func_name);
    if (!func) {
        std::cerr << "  Function not found: " << bench.func_name << " - " << dlerror() << "\n";
        return -1.0;
    }

    // Warmup
    for (int i = 0; i < 3; i++) {
        func();
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count() / iterations;  // ms per iteration
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <shared_library.so>\n";
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load library: " << dlerror() << "\n";
        return 1;
    }

    std::cout << "==================================================================================\n";
    std::cout << "Scatter/Gather Performance Benchmarks\n";
    std::cout << "==================================================================================\n\n";

    Benchmark benchmarks[] = {
        // 1D Gather
        {"1D Gather Small (1K elem, 128 indices)", "bench_gather_1d_small", "1D Gather"},
        {"1D Gather Medium (10K elem, 512 indices)", "bench_gather_1d_medium", "1D Gather"},
        {"1D Gather Large (100K elem, 2K indices)", "bench_gather_1d_large", "1D Gather"},

        // 2D Gather
        {"2D Gather Small (256x128, 32 rows)", "bench_gather_2d_small", "2D Gather"},
        {"2D Gather Medium (1000x512, 64 rows)", "bench_gather_2d_medium", "2D Gather"},

        // 3D Gather
        {"3D Gather Axis 0 (64x64x32, 16 slices)", "bench_gather_3d_axis0", "3D Gather"},

        // 1D Scatter
        {"1D Scatter Small (1K elem, 128 updates)", "bench_scatter_1d_small", "1D Scatter"},
        {"1D Scatter Medium (10K elem, 512 updates)", "bench_scatter_1d_medium", "1D Scatter"},
        {"1D Scatter Large (100K elem, 2K updates)", "bench_scatter_1d_large", "1D Scatter"},

        // 2D Scatter
        {"2D Scatter Small (256x128, 32 rows)", "bench_scatter_2d_small", "2D Scatter"},
        {"2D Scatter Medium (1000x512, 64 rows)", "bench_scatter_2d_medium", "2D Scatter"},

        // Combined
        {"Gather+Scatter Combined (5000x256, 64 ops)", "bench_gather_scatter_combined", "Combined"}
    };

    int numBenchmarks = sizeof(benchmarks) / sizeof(Benchmark);

    // Different iteration counts based on size
    int iterations[] = {
        1000, 500, 100,  // 1D gather
        500, 100,         // 2D gather
        200,              // 3D gather
        1000, 500, 100,  // 1D scatter
        500, 100,         // 2D scatter
        50                // combined
    };

    std::string currentCategory = "";
    std::cout << std::fixed << std::setprecision(4);

    for (int i = 0; i < numBenchmarks; i++) {
        if (currentCategory != benchmarks[i].category) {
            currentCategory = benchmarks[i].category;
            std::cout << "\n" << currentCategory << ":\n";
            std::cout << std::string(80, '-') << "\n";
        }

        double time_ms = runBenchmark(handle, benchmarks[i], iterations[i]);
        if (time_ms >= 0) {
            std::cout << "  " << std::left << std::setw(50) << benchmarks[i].name
                      << std::right << std::setw(10) << time_ms << " ms/iter"
                      << " (" << iterations[i] << " iters)\n";
        }
    }

    std::cout << "\n==================================================================================\n";
    std::cout << "Note: Compare these results with bench_scatter_gather_native for speedup analysis\n";
    std::cout << "==================================================================================\n";

    dlclose(handle);
    return 0;
}
