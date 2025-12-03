/**
 * INT8 MatMul Benchmark Runner
 *
 * Compares SimpLang INT8 matmul against:
 * 1. Scalar reference (for correctness)
 * 2. Eigen INT8 matmul
 * 3. VNNI-optimized reference (when available)
 *
 * Compile:
 *   g++ -O3 -march=native -mavx512vnni -I thirdparty/eigen \
 *       bench_int8_matmul_runner.cpp -o bench_int8_matmul_runner -ldl
 */

#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cstring>
#include <immintrin.h>
#include <Eigen/Dense>

typedef int32_t (*KernelFuncI32)();

// ============================================================================
// Reference Implementations
// ============================================================================

// Scalar reference implementation (for correctness verification)
// Signed × Signed: A and B both use values -64 to 62
template<int N>
int32_t scalar_matmul_int8() {
    std::vector<int8_t> A(N * N);
    std::vector<int8_t> B(N * N);
    std::vector<int32_t> C(N * N, 0);

    // Initialize: both A and B use same values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int val = ((i * N + j) % 127) - 64;
            A[i * N + j] = (int8_t)val;
            B[j * N + i] = (int8_t)val;
        }
    }

    // Compute C = A × B (i8 × i8 → i32)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < N; k++) {
                sum += (int32_t)A[i * N + k] * (int32_t)B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    // Compute checksum
    int32_t checksum = 0;
    for (int i = 0; i < N * N; i++) {
        checksum += C[i];
    }
    return checksum;
}

// Eigen reference implementation (i8 × i8)
template<int N>
int32_t eigen_matmul_int8() {
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> A(N, N);
    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> B(N, N);

    // Initialize
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int val = ((i * N + j) % 127) - 64;
            A(i, j) = (int8_t)val;
            B(j, i) = (int8_t)val;
        }
    }

    // Cast to i32 and multiply
    Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> C =
        (A.cast<int32_t>() * B.cast<int32_t>()).eval();

    return C.sum();
}

// VNNI-optimized reference implementation
// Uses vpmaddwd (i16×i16→i32) for signed int8 multiplication
template<int N>
int32_t vnni_matmul_int8() {
    // Allocate aligned memory on heap for large sizes
    int8_t* A = (int8_t*)aligned_alloc(64, N * N * sizeof(int8_t));
    int8_t* B = (int8_t*)aligned_alloc(64, N * N * sizeof(int8_t));
    int8_t* B_T = (int8_t*)aligned_alloc(64, N * N * sizeof(int8_t));
    int32_t* C = (int32_t*)aligned_alloc(64, N * N * sizeof(int32_t));
    memset(C, 0, N * N * sizeof(int32_t));

    // Initialize: same values for A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int val = ((i * N + j) % 127) - 64;
            A[i * N + j] = (int8_t)val;
            B[j * N + i] = (int8_t)val;
        }
    }

    // Create B_T where B_T[j][k] = B[k][j] for SIMD-friendly row access
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            B_T[j * N + k] = B[k * N + j];
        }
    }

    // Matmul using vpmaddwd for signed i8×i8
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m512i acc = _mm512_setzero_si512();

            // Process K dimension in chunks of 64
            int k = 0;
            for (; k + 63 < N; k += 64) {
                __m512i va = _mm512_loadu_si512(&A[i * N + k]);
                __m512i vb = _mm512_loadu_si512(&B_T[j * N + k]);

                // Sign-extend i8 to i16 for vpmaddwd
                __m512i va_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 0));
                __m512i va_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 1));
                __m512i vb_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 0));
                __m512i vb_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 1));

                // vpmaddwd: multiply i16 pairs and add adjacent
                __m512i prod_lo = _mm512_madd_epi16(va_lo, vb_lo);
                __m512i prod_hi = _mm512_madd_epi16(va_hi, vb_hi);

                acc = _mm512_add_epi32(acc, prod_lo);
                acc = _mm512_add_epi32(acc, prod_hi);
            }

            int32_t sum = _mm512_reduce_add_epi32(acc);

            // Handle remaining elements
            for (; k < N; k++) {
                sum += (int32_t)A[i * N + k] * (int32_t)B_T[j * N + k];
            }

            C[i * N + j] = sum;
        }
    }

    // Compute checksum
    int32_t checksum = 0;
    for (int i = 0; i < N * N; i++) {
        checksum += C[i];
    }
    free(A);
    free(B);
    free(B_T);
    free(C);
    return checksum;
}

// ============================================================================
// Benchmarking Infrastructure
// ============================================================================

// Volatile sink to prevent dead code elimination
volatile int32_t g_sink = 0;

// DoNotOptimize - prevents compiler from optimizing away computation
template<typename T>
inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template<typename Func>
double benchmark(Func func, int iterations) {
    // Warmup - multiple iterations to stabilize CPU frequency and caches
    int32_t result;
    for (int w = 0; w < 5; w++) {
        result = func();
        DoNotOptimize(result);
    }

    // Benchmark - accumulate results to prevent optimization
    int32_t accumulator = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        result = func();
        DoNotOptimize(result);
        accumulator ^= result;  // XOR to prevent optimization
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Store to volatile to ensure computation isn't eliminated
    g_sink = accumulator;

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsed_ms / iterations;
}

struct BenchmarkResult {
    int N;
    double simplang_ms;
    double eigen_ms;
    double vnni_ms;
    double simplang_giops;
    double eigen_giops;
    double vnni_giops;
    int32_t simplang_checksum;
    int32_t eigen_checksum;
    int32_t vnni_checksum;
    int32_t reference_checksum;
    bool correct;
};

template<int N>
BenchmarkResult benchmark_size(void* handle, const char* func_name, int iterations) {
    BenchmarkResult result = {};
    result.N = N;

    // Get SimpLang kernel
    KernelFuncI32 simplang_kernel = (KernelFuncI32)dlsym(handle, func_name);
    if (!simplang_kernel) {
        std::cerr << "Failed to find " << func_name << std::endl;
        return result;
    }

    // Compute reference checksum
    result.reference_checksum = scalar_matmul_int8<N>();

    // Benchmark SimpLang
    result.simplang_ms = benchmark(simplang_kernel, iterations);
    result.simplang_checksum = simplang_kernel();

    // Benchmark Eigen
    result.eigen_ms = benchmark(eigen_matmul_int8<N>, iterations);
    result.eigen_checksum = eigen_matmul_int8<N>();

    // Benchmark VNNI
    result.vnni_ms = benchmark(vnni_matmul_int8<N>, iterations);
    result.vnni_checksum = vnni_matmul_int8<N>();

    // Compute GIOP/s (2*N^3 operations for matmul)
    double giops = 2.0 * N * N * N / 1e9;
    result.simplang_giops = giops / (result.simplang_ms / 1000.0);
    result.eigen_giops = giops / (result.eigen_ms / 1000.0);
    result.vnni_giops = giops / (result.vnni_ms / 1000.0);

    // Check correctness
    result.correct = (result.simplang_checksum == result.reference_checksum) &&
                     (result.eigen_checksum == result.reference_checksum) &&
                     (result.vnni_checksum == result.reference_checksum);

    return result;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::setw(5) << r.N << "×" << std::setw(4) << r.N << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << r.simplang_ms << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << r.simplang_giops << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << r.eigen_ms << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << r.eigen_giops << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << r.vnni_ms << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << r.vnni_giops << " │ ";

    // Ratio vs VNNI
    double ratio_vs_vnni = (r.vnni_ms / r.simplang_ms) * 100.0;
    std::cout << std::setw(7) << std::fixed << std::setprecision(1) << ratio_vs_vnni << "% │";

    if (r.correct) {
        std::cout << " ✓" << std::endl;
    } else {
        std::cout << " ✗" << std::endl;
        std::cout << "       Checksums: SL=" << r.simplang_checksum
                  << " Eigen=" << r.eigen_checksum
                  << " VNNI=" << r.vnni_checksum
                  << " Ref=" << r.reference_checksum << std::endl;
    }
}

int main(int argc, char* argv[]) {
    const char* so_path = "/tmp/bench_int8_matmul.so";
    if (argc > 1) {
        so_path = argv[1];
    }

    std::cout << "═══════════════════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   INT8 GEMM Benchmark: SimpLang vs Eigen vs VNNI Reference" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << "Loading: " << so_path << std::endl;

    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << " Size   │ SimpLang │  GIOP/s  │  Eigen   │  GIOP/s  │   VNNI   │  GIOP/s  │ vs VNNI │ OK" << std::endl;
    std::cout << "────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼─────────┼────" << std::endl;

    // Run benchmarks
    print_result(benchmark_size<32>(handle, "benchmark_int8_matmul_32", 100));
    print_result(benchmark_size<64>(handle, "benchmark_int8_matmul_64", 50));
    print_result(benchmark_size<128>(handle, "benchmark_int8_matmul_128", 20));
    print_result(benchmark_size<256>(handle, "benchmark_int8_matmul_256", 10));
    print_result(benchmark_size<384>(handle, "benchmark_int8_matmul_384", 5));
    print_result(benchmark_size<512>(handle, "benchmark_int8_matmul_512", 3));
    print_result(benchmark_size<768>(handle, "benchmark_int8_matmul_768", 2));
    print_result(benchmark_size<1024>(handle, "benchmark_int8_matmul_1024", 1));
    print_result(benchmark_size<2048>(handle, "benchmark_int8_matmul_2048", 1));

    std::cout << "═══════════════════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  • GIOP/s = Giga Integer Operations per second (higher is better)" << std::endl;
    std::cout << "  • vs VNNI = (VNNI time / SimpLang time) × 100%" << std::endl;
    std::cout << "  • >100% means SimpLang is faster than VNNI reference" << std::endl;
    std::cout << "  • <100% means SimpLang is slower than VNNI reference" << std::endl;
    std::cout << std::endl;

    dlclose(handle);
    return 0;
}
