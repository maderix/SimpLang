#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cmath>
#include <vector>
#include <iomanip>
#include <Eigen/Dense>

typedef float (*KernelFunc)();
typedef int32_t (*KernelFuncI32)();
typedef int64_t (*KernelFuncI64)();

template<int N>
float eigen_matmul() {
    Eigen::MatrixXf A(N, N);
    Eigen::MatrixXf B(N, N);

    // Initialize matrices (same pattern as SimpLang)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = (i * (float)N + j) / (float)(N * N);
            B(i, j) = (j * (float)N + i) / (float)(N * N);
        }
    }

    Eigen::MatrixXf C = A * B;
    return C.sum();
}

template<int N>
int32_t eigen_matmul_i8() {
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> A(N, N);
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> B(N, N);

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = (int8_t)((i * N + j) / N);
            B(i, j) = (int8_t)((j * N + i) / N);
        }
    }

    // Compute matmul and accumulate into i32
    int32_t checksum = 0;
    auto C = A.cast<int32_t>() * B.cast<int32_t>();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C(i, j);
        }
    }
    return checksum;
}

template<int N>
int32_t eigen_matmul_i16() {
    Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic> A(N, N);
    Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic> B(N, N);

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = (int16_t)((i * N + j) / N);
            B(i, j) = (int16_t)((j * N + i) / N);
        }
    }

    // Compute matmul and accumulate into i32
    int32_t checksum = 0;
    auto C = A.cast<int32_t>() * B.cast<int32_t>();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C(i, j);
        }
    }
    return checksum;
}

int32_t eigen_matmul_i32(int N) {
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> A(N, N);
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> B(N, N);

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = (i * N + j) / N;
            B(i, j) = (j * N + i) / N;
        }
    }

    auto C = A * B;
    return C.sum();
}

int64_t eigen_matmul_i64(int N) {
    Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> A(N, N);
    Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic> B(N, N);

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = (i * N + j) / N;
            B(i, j) = (j * N + i) / N;
        }
    }

    auto C = A * B;
    return C.sum();
}

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

void benchmark_size(const char* so_path, const char* func_name, int N, int iterations) {
    // Load SimpLang kernel
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        return;
    }

    KernelFunc simplang_kernel = (KernelFunc)dlsym(handle, func_name);
    if (!simplang_kernel) {
        std::cerr << "Failed to find " << func_name << std::endl;
        dlclose(handle);
        return;
    }

    // Benchmark SimpLang
    double simplang_time = benchmark(simplang_kernel, iterations);
    float simplang_result = simplang_kernel();
    dlclose(handle);

    // Benchmark Eigen
    double eigen_time = 0.0;
    float eigen_result = 0.0f;

    if (N == 64) {
        eigen_time = benchmark(eigen_matmul<64>, iterations);
        eigen_result = eigen_matmul<64>();
    } else if (N == 128) {
        eigen_time = benchmark(eigen_matmul<128>, iterations);
        eigen_result = eigen_matmul<128>();
    } else if (N == 256) {
        eigen_time = benchmark(eigen_matmul<256>, iterations);
        eigen_result = eigen_matmul<256>();
    } else if (N == 512) {
        eigen_time = benchmark(eigen_matmul<512>, iterations);
        eigen_result = eigen_matmul<512>();
    } else if (N == 1024) {
        eigen_time = benchmark(eigen_matmul<1024>, iterations);
        eigen_result = eigen_matmul<1024>();
    }

    // Compute metrics
    double gflops = 2.0 * N * N * N / 1e9;
    double simplang_gflops_per_sec = gflops / (simplang_time / 1000.0);
    double eigen_gflops_per_sec = gflops / (eigen_time / 1000.0);
    double ratio = eigen_time / simplang_time;

    // Display results
    std::cout << std::setw(6) << N << "×" << std::setw(4) << N << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << simplang_time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << simplang_gflops_per_sec << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << eigen_time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << eigen_gflops_per_sec << " │ ";
    std::cout << std::setw(7) << std::fixed << std::setprecision(2) << (ratio * 100.0) << "% │";

    // Correctness check
    if (std::abs(simplang_result - eigen_result) / eigen_result < 0.01) {
        std::cout << " ✓" << std::endl;
    } else {
        std::cout << " ✗ MISMATCH" << std::endl;
    }
}

void benchmark_size_i8(const char* so_path, const char* func_name, int N, int iterations) {
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        return;
    }

    KernelFuncI32 simplang_kernel = (KernelFuncI32)dlsym(handle, func_name);
    if (!simplang_kernel) {
        std::cerr << "Failed to find " << func_name << std::endl;
        dlclose(handle);
        return;
    }

    // Benchmark SimpLang
    double simplang_time = benchmark(simplang_kernel, iterations);
    int32_t simplang_result = simplang_kernel();
    dlclose(handle);

    // Benchmark Eigen
    double eigen_time = benchmark(eigen_matmul_i8<256>, iterations);
    int32_t eigen_result = eigen_matmul_i8<256>();

    // Compute metrics (GIOP for integers)
    double giops = 2.0 * N * N * N / 1e9;
    double simplang_giops_per_sec = giops / (simplang_time / 1000.0);
    double eigen_giops_per_sec = giops / (eigen_time / 1000.0);
    double ratio = eigen_time / simplang_time;

    // Display results
    std::cout << " i8  " << std::setw(3) << N << "×" << std::setw(4) << N << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << simplang_time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << simplang_giops_per_sec << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << eigen_time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << eigen_giops_per_sec << " │ ";
    std::cout << std::setw(7) << std::fixed << std::setprecision(2) << (ratio * 100.0) << "% │";

    // Correctness check
    if (simplang_result == eigen_result) {
        std::cout << " ✓" << std::endl;
    } else {
        std::cout << " ✗ MISMATCH (SL: " << simplang_result << ", Eigen: " << eigen_result << ")" << std::endl;
    }
}

void benchmark_size_i16(const char* so_path, const char* func_name, int N, int iterations) {
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        return;
    }

    KernelFuncI32 simplang_kernel = (KernelFuncI32)dlsym(handle, func_name);
    if (!simplang_kernel) {
        std::cerr << "Failed to find " << func_name << std::endl;
        dlclose(handle);
        return;
    }

    // Benchmark SimpLang
    double simplang_time = benchmark(simplang_kernel, iterations);
    int32_t simplang_result = simplang_kernel();
    dlclose(handle);

    // Benchmark Eigen
    double eigen_time = benchmark(eigen_matmul_i16<256>, iterations);
    int32_t eigen_result = eigen_matmul_i16<256>();

    // Compute metrics (GIOP for integers)
    double giops = 2.0 * N * N * N / 1e9;
    double simplang_giops_per_sec = giops / (simplang_time / 1000.0);
    double eigen_giops_per_sec = giops / (eigen_time / 1000.0);
    double ratio = eigen_time / simplang_time;

    // Display results
    std::cout << " i16 " << std::setw(3) << N << "×" << std::setw(4) << N << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << simplang_time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << simplang_giops_per_sec << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << eigen_time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << eigen_giops_per_sec << " │ ";
    std::cout << std::setw(7) << std::fixed << std::setprecision(2) << (ratio * 100.0) << "% │";

    // Correctness check
    if (simplang_result == eigen_result) {
        std::cout << " ✓" << std::endl;
    } else {
        std::cout << " ✗ MISMATCH (SL: " << simplang_result << ", Eigen: " << eigen_result << ")" << std::endl;
    }
}

void benchmark_size_i32(const char* so_path, const char* func_name, int N, int iterations) {
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        return;
    }

    KernelFuncI32 simplang_kernel = (KernelFuncI32)dlsym(handle, func_name);
    if (!simplang_kernel) {
        std::cerr << "Failed to find " << func_name << std::endl;
        dlclose(handle);
        return;
    }

    // Benchmark SimpLang
    double simplang_time = benchmark(simplang_kernel, iterations);
    int32_t simplang_result = simplang_kernel();
    dlclose(handle);

    // Benchmark Eigen
    auto eigen_func = [N]() { return eigen_matmul_i32(N); };
    double eigen_time = benchmark(eigen_func, iterations);
    int32_t eigen_result = eigen_matmul_i32(N);

    // Compute metrics (GIOP for integers)
    double giops = 2.0 * N * N * N / 1e9;
    double simplang_giops_per_sec = giops / (simplang_time / 1000.0);
    double eigen_giops_per_sec = giops / (eigen_time / 1000.0);
    double ratio = eigen_time / simplang_time;

    // Display results
    std::cout << " i32 " << std::setw(3) << N << "×" << std::setw(4) << N << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << simplang_time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << simplang_giops_per_sec << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << eigen_time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << eigen_giops_per_sec << " │ ";
    std::cout << std::setw(7) << std::fixed << std::setprecision(2) << (ratio * 100.0) << "% │";

    // Correctness check
    if (simplang_result == eigen_result) {
        std::cout << " ✓" << std::endl;
    } else {
        std::cout << " ✗ MISMATCH (SL: " << simplang_result << ", Eigen: " << eigen_result << ")" << std::endl;
    }
}

void benchmark_size_i64(const char* so_path, const char* func_name, int N, int iterations) {
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        return;
    }

    KernelFuncI64 simplang_kernel = (KernelFuncI64)dlsym(handle, func_name);
    if (!simplang_kernel) {
        std::cerr << "Failed to find " << func_name << std::endl;
        dlclose(handle);
        return;
    }

    // Benchmark SimpLang
    double simplang_time = benchmark(simplang_kernel, iterations);
    int64_t simplang_result = simplang_kernel();
    dlclose(handle);

    // Benchmark Eigen
    auto eigen_func = [N]() { return eigen_matmul_i64(N); };
    double eigen_time = benchmark(eigen_func, iterations);
    int64_t eigen_result = eigen_matmul_i64(N);

    // Compute metrics (GIOP for integers)
    double giops = 2.0 * N * N * N / 1e9;
    double simplang_giops_per_sec = giops / (simplang_time / 1000.0);
    double eigen_giops_per_sec = giops / (eigen_time / 1000.0);
    double ratio = eigen_time / simplang_time;

    // Display results
    std::cout << " i64 " << std::setw(3) << N << "×" << std::setw(4) << N << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << simplang_time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << simplang_giops_per_sec << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << eigen_time << " ms │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << eigen_giops_per_sec << " │ ";
    std::cout << std::setw(7) << std::fixed << std::setprecision(2) << (ratio * 100.0) << "% │";

    // Correctness check
    if (simplang_result == eigen_result) {
        std::cout << " ✓" << std::endl;
    } else {
        std::cout << " ✗ MISMATCH (SL: " << simplang_result << ", Eigen: " << eigen_result << ")" << std::endl;
    }
}

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   GEMM Performance Benchmark: SimpLang vs Eigen (Multiple Sizes)" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;

    // Table header
    std::cout << " Size    │ SimpLang │  GFLOP/s │  Eigen   │  GFLOP/s │ Ratio   │ Status" << std::endl;
    std::cout << "─────────┼──────────┼──────────┼──────────┼──────────┼─────────┼────────" << std::endl;

    // Benchmark different sizes
    benchmark_size("/tmp/bench_matmul_64.so", "benchmark_matmul_64", 64, 10);
    benchmark_size("/tmp/bench_matmul_128.so", "benchmark_matmul_128", 128, 10);
    benchmark_size("/tmp/bench_matmul_256.so", "benchmark_matmul_256", 256, 5);
    benchmark_size("/tmp/bench_matmul_512.so", "benchmark_matmul_512", 512, 3);
    benchmark_size("/tmp/bench_matmul_1024.so", "benchmark_matmul_1024", 1024, 2);

    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << "Integer Types (i32 - Multiple Sizes):" << std::endl;
    std::cout << "─────────┼──────────┼──────────┼──────────┼──────────┼─────────┼────────" << std::endl;

    benchmark_size_i32("/tmp/bench_all_sizes.so", "benchmark_matmul_64_i32", 64, 10);
    benchmark_size_i32("/tmp/bench_all_sizes.so", "benchmark_matmul_128_i32", 128, 10);
    benchmark_size_i32("/tmp/bench_all_sizes.so", "benchmark_matmul_256_i32", 256, 5);
    benchmark_size_i32("/tmp/bench_all_sizes.so", "benchmark_matmul_512_i32", 512, 3);
    benchmark_size_i32("/tmp/bench_all_sizes.so", "benchmark_matmul_1024_i32", 1024, 2);

    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << "i64 Type (256×256):" << std::endl;
    std::cout << "─────────┼──────────┼──────────┼──────────┼──────────┼─────────┼────────" << std::endl;

    benchmark_size_i64("/tmp/bench_all_sizes.so", "benchmark_matmul_256_i64", 256, 5);

    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << "Notes:" << std::endl;
    std::cout << "• Ratio = (Eigen time / SimpLang time) × 100%" << std::endl;
    std::cout << "• >100% means SimpLang is slower than Eigen" << std::endl;
    std::cout << "• <100% means SimpLang is faster than Eigen" << std::endl;
    std::cout << "• 70-90% is competitive for a DSL compiler" << std::endl;
    std::cout << std::endl;

    return 0;
}
