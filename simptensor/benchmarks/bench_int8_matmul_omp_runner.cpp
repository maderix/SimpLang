/**
 * INT8 MatMul OpenMP Runner
 *
 * Calls SimpLang partial matmul functions in parallel using OpenMP.
 * Compares with single-threaded full matmul.
 *
 * Compile:
 *   g++ -O3 -march=native -fopenmp bench_int8_matmul_omp_runner.cpp -o /tmp/bench_int8_omp -ldl
 *
 * Run:
 *   OMP_NUM_THREADS=8 /tmp/bench_int8_omp /tmp/bench_int8_partial.so
 */

#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <vector>
#include <cstring>
#include <iomanip>
#include <omp.h>

// SimpLang function signatures
typedef int32_t (*FullMatmulFunc)();
typedef int32_t (*ChunkMatmulFunc)(int8_t*, int8_t*, int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t);

const int N = 1024;

volatile int32_t g_sink = 0;

template<typename T>
inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

// Reference scalar for verification
int32_t scalar_matmul_checksum() {
    std::vector<int8_t> A(N * N);
    std::vector<int8_t> B(N * N);
    std::vector<int32_t> C(N * N, 0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int val = ((i * N + j) % 127) - 64;
            A[i * N + j] = (int8_t)val;
            B[j * N + i] = (int8_t)val;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < N; k++) {
                sum += (int32_t)A[i * N + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    int32_t checksum = 0;
    for (int i = 0; i < N * N; i++) {
        checksum += C[i];
    }
    return checksum;
}

int main(int argc, char* argv[]) {
    const char* so_path = "/tmp/bench_int8_partial.so";
    if (argc > 1) {
        so_path = argv[1];
    }

    int num_threads = omp_get_max_threads();

    std::cout << "=============================================================================" << std::endl;
    std::cout << "   INT8 MatMul OpenMP Parallelization Test" << std::endl;
    std::cout << "=============================================================================" << std::endl;
    std::cout << "OpenMP threads: " << num_threads << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << std::endl;

    // Load shared library
    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        std::cerr << "Compile first: cd build_mlir && ./src/simplang ../simptensor/benchmarks/bench_int8_matmul_partial.sl --emit-mlir --llvm-vectorize -o /tmp/bench_int8_partial.o && gcc -shared -o /tmp/bench_int8_partial.so /tmp/bench_int8_partial.o -lm" << std::endl;
        return 1;
    }

    // Load functions
    FullMatmulFunc full_matmul = (FullMatmulFunc)dlsym(handle, "int8_matmul_full_1024");
    ChunkMatmulFunc chunk_matmul = (ChunkMatmulFunc)dlsym(handle, "int8_matmul_chunk_1024");

    if (!full_matmul) {
        std::cerr << "Failed to find int8_matmul_full_1024" << std::endl;
        return 1;
    }

    // Calculate reference checksum
    int32_t ref_checksum = scalar_matmul_checksum();
    std::cout << "Reference checksum: " << ref_checksum << std::endl;

    // =========================================================================
    // Test 1: Single-threaded full matmul (uses VNNI)
    // =========================================================================
    std::cout << "\n--- Single-threaded Full MatMul (VNNI optimized) ---" << std::endl;

    // Warmup
    for (int w = 0; w < 3; w++) {
        int32_t r = full_matmul();
        DoNotOptimize(r);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 5;
    int32_t checksum = 0;
    for (int i = 0; i < iterations; i++) {
        checksum = full_matmul();
        DoNotOptimize(checksum);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double single_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double single_giops = (2.0 * N * N * N / 1e9) / (single_ms / 1000.0);

    std::cout << "Time: " << std::fixed << std::setprecision(3) << single_ms << " ms" << std::endl;
    std::cout << "GIOP/s: " << std::fixed << std::setprecision(2) << single_giops << std::endl;
    std::cout << "Checksum: " << checksum << (checksum == ref_checksum ? " ✓" : " ✗") << std::endl;

    // =========================================================================
    // Test 2: Parallel chunk matmul (if available)
    // =========================================================================
    if (chunk_matmul) {
        std::cout << "\n--- Parallel Chunk MatMul (" << num_threads << " threads) ---" << std::endl;

        // Allocate matrices
        std::vector<int8_t> A(N * N);
        std::vector<int8_t> B(N * N);
        std::vector<int32_t> C(N * N, 0);

        // Initialize same as SimpLang
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int val = ((i * N + j) % 127) - 64;
                A[i * N + j] = (int8_t)val;
                B[j * N + i] = (int8_t)val;
            }
        }

        int chunk_size = N / num_threads;

        // Warmup
        for (int w = 0; w < 2; w++) {
            std::fill(C.begin(), C.end(), 0);
            #pragma omp parallel for
            for (int t = 0; t < num_threads; t++) {
                int64_t row_start = t * chunk_size;
                chunk_matmul(A.data(), B.data(), C.data(),
                             row_start, chunk_size, N, N, N);
            }
        }

        // Benchmark
        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; iter++) {
            std::fill(C.begin(), C.end(), 0);
            #pragma omp parallel for
            for (int t = 0; t < num_threads; t++) {
                int64_t row_start = t * chunk_size;
                chunk_matmul(A.data(), B.data(), C.data(),
                             row_start, chunk_size, N, N, N);
            }
            DoNotOptimize(C[0]);
        }
        end = std::chrono::high_resolution_clock::now();

        double parallel_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        double parallel_giops = (2.0 * N * N * N / 1e9) / (parallel_ms / 1000.0);

        // Compute checksum
        int32_t par_checksum = 0;
        for (int i = 0; i < N * N; i++) {
            par_checksum += C[i];
        }

        std::cout << "Time: " << std::fixed << std::setprecision(3) << parallel_ms << " ms" << std::endl;
        std::cout << "GIOP/s: " << std::fixed << std::setprecision(2) << parallel_giops << std::endl;
        std::cout << "Checksum: " << par_checksum << (par_checksum == ref_checksum ? " ✓" : " ✗") << std::endl;
        std::cout << "Speedup vs single: " << std::fixed << std::setprecision(2) << (single_ms / parallel_ms) << "x" << std::endl;
    } else {
        std::cout << "\nChunk matmul function not found - skipping parallel test" << std::endl;
    }

    std::cout << "\n=============================================================================" << std::endl;

    dlclose(handle);
    return 0;
}
