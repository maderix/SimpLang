#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cmath>
#include <vector>
#include <iomanip>
#include <string>

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

typedef float (*KernelFunc)();

struct BenchmarkResult {
    std::string name;
    double time_ms;
    double gflops;
    float checksum;
    bool valid;
};

template<typename Func>
double benchmark(Func func, int iterations) {
    // Warmup
    func();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsed_ms / iterations;
}

BenchmarkResult run_benchmark(void* handle, const char* func_name, int N, int iterations) {
    BenchmarkResult result;
    result.name = func_name;
    result.valid = false;

    KernelFunc kernel = (KernelFunc)dlsym(handle, func_name);
    if (!kernel) {
        std::cerr << "Failed to find " << func_name << std::endl;
        return result;
    }

    result.time_ms = benchmark(kernel, iterations);
    result.checksum = kernel();

    // GFLOP/s for NxN matmul: 2*N^3 operations
    double gflops = 2.0 * N * N * N / 1e9;
    result.gflops = gflops / (result.time_ms / 1000.0);
    result.valid = true;

    return result;
}

#ifdef USE_EIGEN
template<int N>
float eigen_matmul() {
    Eigen::MatrixXf A(N, N);
    Eigen::MatrixXf B(N, N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = (i * (float)N + j) / (float)(N * N);
            B(i, j) = (j * (float)N + i) / (float)(N * N);
        }
    }

    Eigen::MatrixXf C = A * B;
    return C.sum();
}
#endif

void print_separator() {
    std::cout << "─────────────────────────────────────────────────────────────────────────" << std::endl;
}

void print_header() {
    std::cout << std::left << std::setw(35) << " Variant"
              << " │ " << std::right << std::setw(10) << "Time (ms)"
              << " │ " << std::setw(10) << "GFLOP/s"
              << " │ " << std::setw(8) << "Speedup"
              << " │ Status" << std::endl;
    print_separator();
}

int main(int argc, char* argv[]) {
    const char* so_path = "/tmp/bench_f32_annotated.so";
    if (argc > 1) {
        so_path = argv[1];
    }

    std::cout << "═══════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   F32 Matmul Benchmark: Annotation Optimization Comparison" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;

    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        return 1;
    }

    // 1024x1024 benchmarks
    std::cout << "1024×1024 F32 Matmul (2.15 GFLOP per call):" << std::endl;
    print_separator();
    print_header();

    std::vector<std::pair<std::string, std::string>> benchmarks_1024 = {
        {"matmul_1024_baseline", "Baseline (16×16×16)"},
        {"matmul_1024_tile32", "@tile(32,32,32)"},
        {"matmul_1024_tile64", "@tile(64,64,64)"},
        {"matmul_1024_tile128", "@tile(128,128,128)"},
        {"matmul_1024_tile256", "@tile(256,256,256)"},
        {"matmul_1024_parallel_tile64", "@parallel @tile(64,64,64)"},
        {"matmul_1024_parallel_tile128", "@parallel @tile(128,128,128)"},
        {"matmul_1024_tile64_prefetch", "@tile(64,64,64) @prefetch(4)"},
        {"matmul_1024_full_opt", "@parallel @tile(64) @prefetch @unroll"},
        {"matmul_1024_tile64_vec256", "@tile(64,64,64) @vectorize(256)"},
    };

    BenchmarkResult baseline_1024;
    std::vector<BenchmarkResult> results_1024;

    for (size_t i = 0; i < benchmarks_1024.size(); i++) {
        auto& [func_name, display_name] = benchmarks_1024[i];
        BenchmarkResult r = run_benchmark(handle, func_name.c_str(), 1024, 3);
        r.name = display_name;

        if (i == 0) baseline_1024 = r;
        results_1024.push_back(r);

        if (r.valid) {
            double speedup = baseline_1024.time_ms / r.time_ms;
            bool correct = std::abs(r.checksum - baseline_1024.checksum) / baseline_1024.checksum < 0.01;

            std::cout << std::left << std::setw(35) << (" " + r.name)
                      << " │ " << std::right << std::setw(10) << std::fixed << std::setprecision(3) << r.time_ms
                      << " │ " << std::setw(10) << std::fixed << std::setprecision(2) << r.gflops
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(2) << speedup << "x"
                      << " │ " << (correct ? "✓" : "✗ MISMATCH") << std::endl;
        }
    }

    // 512x512 benchmarks
    std::cout << std::endl;
    std::cout << "512×512 F32 Matmul (0.27 GFLOP per call):" << std::endl;
    print_separator();
    print_header();

    std::vector<std::pair<std::string, std::string>> benchmarks_512 = {
        {"matmul_512_baseline", "Baseline (16×16×16)"},
        {"matmul_512_tile64", "@tile(64,64,64)"},
        {"matmul_512_parallel_tile64", "@parallel @tile(64,64,64)"},
    };

    BenchmarkResult baseline_512;
    for (size_t i = 0; i < benchmarks_512.size(); i++) {
        auto& [func_name, display_name] = benchmarks_512[i];
        BenchmarkResult r = run_benchmark(handle, func_name.c_str(), 512, 5);
        r.name = display_name;

        if (i == 0) baseline_512 = r;

        if (r.valid) {
            double speedup = baseline_512.time_ms / r.time_ms;
            bool correct = std::abs(r.checksum - baseline_512.checksum) / baseline_512.checksum < 0.01;

            std::cout << std::left << std::setw(35) << (" " + r.name)
                      << " │ " << std::right << std::setw(10) << std::fixed << std::setprecision(3) << r.time_ms
                      << " │ " << std::setw(10) << std::fixed << std::setprecision(2) << r.gflops
                      << " │ " << std::setw(7) << std::fixed << std::setprecision(2) << speedup << "x"
                      << " │ " << (correct ? "✓" : "✗ MISMATCH") << std::endl;
        }
    }

#ifdef USE_EIGEN
    // Eigen comparison
    std::cout << std::endl;
    std::cout << "Eigen Comparison:" << std::endl;
    print_separator();

    double eigen_1024_time = benchmark(eigen_matmul<1024>, 3);
    double eigen_1024_gflops = (2.0 * 1024 * 1024 * 1024 / 1e9) / (eigen_1024_time / 1000.0);

    double eigen_512_time = benchmark(eigen_matmul<512>, 5);
    double eigen_512_gflops = (2.0 * 512 * 512 * 512 / 1e9) / (eigen_512_time / 1000.0);

    std::cout << " Eigen 1024×1024: " << std::fixed << std::setprecision(2) << eigen_1024_gflops << " GFLOP/s" << std::endl;
    std::cout << " Eigen 512×512:   " << std::fixed << std::setprecision(2) << eigen_512_gflops << " GFLOP/s" << std::endl;

    // Find best SimpLang result
    double best_1024_gflops = 0;
    std::string best_1024_name;
    for (const auto& r : results_1024) {
        if (r.valid && r.gflops > best_1024_gflops) {
            best_1024_gflops = r.gflops;
            best_1024_name = r.name;
        }
    }

    std::cout << std::endl;
    std::cout << " Best SimpLang 1024×1024: " << best_1024_name << " @ "
              << std::fixed << std::setprecision(2) << best_1024_gflops << " GFLOP/s" << std::endl;
    std::cout << " Ratio vs Eigen: " << std::fixed << std::setprecision(1)
              << (best_1024_gflops / eigen_1024_gflops * 100.0) << "%" << std::endl;
#endif

    std::cout << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "Notes:" << std::endl;
    std::cout << "• Speedup is relative to baseline (default 16×16×16 tiling)" << std::endl;
    std::cout << "• @parallel uses scf.parallel for OpenMP-style parallelization" << std::endl;
    std::cout << "• @prefetch(N) inserts memory prefetch hints N iterations ahead" << std::endl;
    std::cout << "• @unroll(N) unrolls innermost loops by factor N" << std::endl;
    std::cout << "• @vectorize(W) hints preferred SIMD width (128/256/512 bits)" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════" << std::endl;

    dlclose(handle);
    return 0;
}
