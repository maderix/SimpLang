// Host runner for multi-dimensional array benchmark
#include <iostream>
#include <dlfcn.h>
#include <chrono>
#include <iomanip>
#include <cmath>

// C++ baseline function declarations
extern "C" {
    double bench_1d_baseline_cpp();
    double bench_2d_indexed_cpp();
    double bench_2d_flattened_cpp();
    double bench_3d_indexed_cpp();
    int64_t bench_2d_int_indexed_cpp();
    int64_t bench_2d_int_flattened_cpp();
    double bench_2d_large_indexed_cpp();
    double bench_2d_large_flattened_cpp();
}

typedef double (*BenchFuncF64)();
typedef int64_t (*BenchFuncI64)();

double benchmark_f64(BenchFuncF64 func, const char* name, int iterations = 100) {
    // Warmup
    for (int i = 0; i < 5; i++) {
        func();
    }

    // Actual benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_us = (elapsed_ms * 1000.0) / iterations;

    return avg_us;
}

double benchmark_i64(BenchFuncI64 func, const char* name, int iterations = 100) {
    // Warmup
    for (int i = 0; i < 5; i++) {
        func();
    }

    // Actual benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_us = (elapsed_ms * 1000.0) / iterations;

    return avg_us;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    // Load compiled kernel
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load kernel: " << dlerror() << std::endl;
        return 1;
    }

    // Load all SimpLang benchmark functions
    auto bench_1d = (BenchFuncF64)dlsym(handle, "bench_1d_baseline");
    auto bench_2d_indexed = (BenchFuncF64)dlsym(handle, "bench_2d_indexed");
    auto bench_2d_flattened = (BenchFuncF64)dlsym(handle, "bench_2d_flattened");
    auto bench_3d_indexed = (BenchFuncF64)dlsym(handle, "bench_3d_indexed");
    auto bench_2d_int_indexed = (BenchFuncI64)dlsym(handle, "bench_2d_int_indexed");
    auto bench_2d_int_flattened = (BenchFuncI64)dlsym(handle, "bench_2d_int_flattened");
    auto bench_2d_large_indexed = (BenchFuncF64)dlsym(handle, "bench_2d_large_indexed");
    auto bench_2d_large_flattened = (BenchFuncF64)dlsym(handle, "bench_2d_large_flattened");

    if (!bench_1d || !bench_2d_indexed || !bench_2d_flattened || !bench_3d_indexed ||
        !bench_2d_int_indexed || !bench_2d_int_flattened ||
        !bench_2d_large_indexed || !bench_2d_large_flattened) {
        std::cerr << "Failed to find benchmark functions" << std::endl;
        dlclose(handle);
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Multi-Dimensional Array Benchmark" << std::endl;
    std::cout << "SimpLang (MLIR) vs C++ Baseline (O3)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // ========== SMALL FLOAT ARRAYS (cache-friendly) ==========

    std::cout << "SMALL FLOAT ARRAYS (1K-1K elements, L1 cache)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    double t_1d_sl = benchmark_f64(bench_1d, "1D Baseline (SimpLang)", 100);
    double t_1d_cpp = benchmark_f64(bench_1d_baseline_cpp, "1D Baseline (C++)", 100);

    double t_2d_idx_sl = benchmark_f64(bench_2d_indexed, "2D Indexed (SimpLang)", 100);
    double t_2d_idx_cpp = benchmark_f64(bench_2d_indexed_cpp, "2D Indexed (C++)", 100);

    double t_2d_flat_sl = benchmark_f64(bench_2d_flattened, "2D Flattened (SimpLang)", 100);
    double t_2d_flat_cpp = benchmark_f64(bench_2d_flattened_cpp, "2D Flattened (C++)", 100);

    double t_3d_sl = benchmark_f64(bench_3d_indexed, "3D Indexed (SimpLang)", 100);
    double t_3d_cpp = benchmark_f64(bench_3d_indexed_cpp, "3D Indexed (C++)", 100);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  1D Baseline (1000):     SimpLang: " << std::setw(7) << t_1d_sl << " μs";
    std::cout << "  |  C++: " << std::setw(7) << t_1d_cpp << " μs";
    std::cout << "  |  Ratio: " << std::setprecision(3) << (t_1d_sl / t_1d_cpp) << "x" << std::endl;

    std::cout << std::setprecision(2);
    std::cout << "  2D Indexed (32×32):     SimpLang: " << std::setw(7) << t_2d_idx_sl << " μs";
    std::cout << "  |  C++: " << std::setw(7) << t_2d_idx_cpp << " μs";
    std::cout << "  |  Ratio: " << std::setprecision(3) << (t_2d_idx_sl / t_2d_idx_cpp) << "x" << std::endl;

    std::cout << std::setprecision(2);
    std::cout << "  2D Flattened (32×32):   SimpLang: " << std::setw(7) << t_2d_flat_sl << " μs";
    std::cout << "  |  C++: " << std::setw(7) << t_2d_flat_cpp << " μs";
    std::cout << "  |  Ratio: " << std::setprecision(3) << (t_2d_flat_sl / t_2d_flat_cpp) << "x" << std::endl;

    std::cout << std::setprecision(2);
    std::cout << "  3D Indexed (16×16×4):   SimpLang: " << std::setw(7) << t_3d_sl << " μs";
    std::cout << "  |  C++: " << std::setw(7) << t_3d_cpp << " μs";
    std::cout << "  |  Ratio: " << std::setprecision(3) << (t_3d_sl / t_3d_cpp) << "x" << std::endl;
    std::cout << std::endl;

    // ========== INTEGER ARRAYS (type promotion test) ==========

    std::cout << "INTEGER ARRAYS (256×256 = 64K elements, L2 cache)" << std::endl;
    std::cout << "Testing i32 -> i64 type promotion" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    double t_2d_int_idx_sl = benchmark_i64(bench_2d_int_indexed, "2D Int Indexed (SimpLang)", 100);
    double t_2d_int_idx_cpp = benchmark_i64(bench_2d_int_indexed_cpp, "2D Int Indexed (C++)", 100);

    double t_2d_int_flat_sl = benchmark_i64(bench_2d_int_flattened, "2D Int Flattened (SimpLang)", 100);
    double t_2d_int_flat_cpp = benchmark_i64(bench_2d_int_flattened_cpp, "2D Int Flattened (C++)", 100);

    std::cout << std::setprecision(2);
    std::cout << "  2D Int Indexed:         SimpLang: " << std::setw(7) << t_2d_int_idx_sl << " μs";
    std::cout << "  |  C++: " << std::setw(7) << t_2d_int_idx_cpp << " μs";
    std::cout << "  |  Ratio: " << std::setprecision(3) << (t_2d_int_idx_sl / t_2d_int_idx_cpp) << "x" << std::endl;

    std::cout << std::setprecision(2);
    std::cout << "  2D Int Flattened:       SimpLang: " << std::setw(7) << t_2d_int_flat_sl << " μs";
    std::cout << "  |  C++: " << std::setw(7) << t_2d_int_flat_cpp << " μs";
    std::cout << "  |  Ratio: " << std::setprecision(3) << (t_2d_int_flat_sl / t_2d_int_flat_cpp) << "x" << std::endl;

    std::cout << "  Indexed vs Flattened:   SimpLang: " << std::setprecision(3) << (t_2d_int_idx_sl / t_2d_int_flat_sl) << "x";
    std::cout << "  |  C++: " << (t_2d_int_idx_cpp / t_2d_int_flat_cpp) << "x" << std::endl;
    std::cout << std::endl;

    // ========== LARGE FLOAT ARRAYS (cache pressure) ==========

    std::cout << "LARGE FLOAT ARRAYS (512×512 = 256K elements, L3 cache)" << std::endl;
    std::cout << "Stressing memory hierarchy" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    double t_2d_large_idx_sl = benchmark_f64(bench_2d_large_indexed, "2D Large Indexed (SimpLang)", 20);
    double t_2d_large_idx_cpp = benchmark_f64(bench_2d_large_indexed_cpp, "2D Large Indexed (C++)", 20);

    double t_2d_large_flat_sl = benchmark_f64(bench_2d_large_flattened, "2D Large Flattened (SimpLang)", 20);
    double t_2d_large_flat_cpp = benchmark_f64(bench_2d_large_flattened_cpp, "2D Large Flattened (C++)", 20);

    std::cout << std::setprecision(2);
    std::cout << "  2D Large Indexed:       SimpLang: " << std::setw(7) << t_2d_large_idx_sl << " μs";
    std::cout << "  |  C++: " << std::setw(7) << t_2d_large_idx_cpp << " μs";
    std::cout << "  |  Ratio: " << std::setprecision(3) << (t_2d_large_idx_sl / t_2d_large_idx_cpp) << "x" << std::endl;

    std::cout << std::setprecision(2);
    std::cout << "  2D Large Flattened:     SimpLang: " << std::setw(7) << t_2d_large_flat_sl << " μs";
    std::cout << "  |  C++: " << std::setw(7) << t_2d_large_flat_cpp << " μs";
    std::cout << "  |  Ratio: " << std::setprecision(3) << (t_2d_large_flat_sl / t_2d_large_flat_cpp) << "x" << std::endl;

    std::cout << "  Indexed vs Flattened:   SimpLang: " << std::setprecision(3) << (t_2d_large_idx_sl / t_2d_large_flat_sl) << "x";
    std::cout << "  |  C++: " << (t_2d_large_idx_cpp / t_2d_large_flat_cpp) << "x" << std::endl;
    std::cout << std::endl;

    // ========== CORRECTNESS CHECKS ==========

    std::cout << "========================================" << std::endl;
    std::cout << "Correctness Checks" << std::endl;
    std::cout << "========================================" << std::endl;

    double r1_sl = bench_1d();
    double r2_sl = bench_2d_indexed();
    double r3_sl = bench_2d_flattened();
    double r4_sl = bench_3d_indexed();
    int64_t r5_sl = bench_2d_int_indexed();
    int64_t r6_sl = bench_2d_int_flattened();
    double r7_sl = bench_2d_large_indexed();
    double r8_sl = bench_2d_large_flattened();

    double r1_cpp = bench_1d_baseline_cpp();
    double r2_cpp = bench_2d_indexed_cpp();
    double r3_cpp = bench_2d_flattened_cpp();
    double r4_cpp = bench_3d_indexed_cpp();
    int64_t r5_cpp = bench_2d_int_indexed_cpp();
    int64_t r6_cpp = bench_2d_int_flattened_cpp();
    double r7_cpp = bench_2d_large_indexed_cpp();
    double r8_cpp = bench_2d_large_flattened_cpp();

    bool correct = true;

    std::cout << std::fixed << std::setprecision(1);

    auto check_f64 = [&](double sl, double cpp, double expected, const char* name) {
        bool ok = (std::abs(sl - expected) < 1.0) && (std::abs(cpp - expected) < 1.0);
        std::cout << "  " << std::setw(30) << std::left << name;
        std::cout << "SimpLang: " << std::setw(15) << sl;
        std::cout << "C++: " << std::setw(15) << cpp;
        std::cout << (ok ? "✓" : "✗") << std::endl;
        correct = correct && ok;
    };

    auto check_i64 = [&](int64_t sl, int64_t cpp, int64_t expected, const char* name) {
        bool ok = (sl == expected) && (cpp == expected);
        std::cout << "  " << std::setw(30) << std::left << name;
        std::cout << "SimpLang: " << std::setw(15) << sl;
        std::cout << "C++: " << std::setw(15) << cpp;
        std::cout << (ok ? "✓" : "✗") << std::endl;
        correct = correct && ok;
    };

    check_f64(r1_sl, r1_cpp, 499500.0, "1D baseline:");
    check_f64(r2_sl, r2_cpp, 523776.0, "2D indexed:");
    check_f64(r3_sl, r3_cpp, 523776.0, "2D flattened:");
    check_f64(r4_sl, r4_cpp, 523776.0, "3D indexed:");
    check_i64(r5_sl, r5_cpp, 2147450880, "2D int indexed:");
    check_i64(r6_sl, r6_cpp, 2147450880, "2D int flattened:");
    check_f64(r7_sl, r7_cpp, 34359607296.0, "2D large indexed:");
    check_f64(r8_sl, r8_cpp, 34359607296.0, "2D large flattened:");

    std::cout << std::endl;
    std::cout << "Overall Status: " << (correct ? "PASSED ✓" : "FAILED ✗") << std::endl;

    dlclose(handle);
    return correct ? 0 : 1;
}
