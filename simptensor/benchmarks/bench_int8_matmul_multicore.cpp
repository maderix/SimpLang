/**
 * INT8 MatMul Multicore Benchmark
 *
 * Tests OpenMP parallelization of VNNI INT8 matmul to find
 * the achievable ceiling with multicore on our CPU.
 *
 * The approach: parallelize the outer I loop (row processing)
 * since each row is independent.
 *
 * Compile:
 *   g++ -O3 -march=native -mavx512vnni -fopenmp \
 *       bench_int8_matmul_multicore.cpp -o /tmp/bench_int8_multicore -ldl
 *
 * Run:
 *   OMP_NUM_THREADS=8 /tmp/bench_int8_multicore
 */

#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

typedef int32_t (*KernelFuncI32)();

volatile int32_t g_sink = 0;

template<typename T>
inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

// ============================================================================
// Single-threaded VNNI matmul (baseline)
// ============================================================================
template<int N>
void vnni_matmul_single_thread(const int8_t* A, const int8_t* B_T, int32_t* C) {
    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);

    for (int i = 0; i < N; i += 4) {
        for (int j = 0; j < N; j++) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i bias_acc = _mm512_setzero_si512();

            for (int k = 0; k + 64 <= N; k += 64) {
                __m512i vb = _mm512_loadu_si512(&B_T[j * N + k]);

                __m512i va0 = _mm512_xor_si512(_mm512_loadu_si512(&A[(i+0) * N + k]), sign_flip);
                __m512i va1 = _mm512_xor_si512(_mm512_loadu_si512(&A[(i+1) * N + k]), sign_flip);
                __m512i va2 = _mm512_xor_si512(_mm512_loadu_si512(&A[(i+2) * N + k]), sign_flip);
                __m512i va3 = _mm512_xor_si512(_mm512_loadu_si512(&A[(i+3) * N + k]), sign_flip);

                acc0 = _mm512_dpbusd_epi32(acc0, va0, vb);
                acc1 = _mm512_dpbusd_epi32(acc1, va1, vb);
                acc2 = _mm512_dpbusd_epi32(acc2, va2, vb);
                acc3 = _mm512_dpbusd_epi32(acc3, va3, vb);

                bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, vb);
            }

            int32_t sum0 = _mm512_reduce_add_epi32(acc0);
            int32_t sum1 = _mm512_reduce_add_epi32(acc1);
            int32_t sum2 = _mm512_reduce_add_epi32(acc2);
            int32_t sum3 = _mm512_reduce_add_epi32(acc3);
            int32_t bias = _mm512_reduce_add_epi32(bias_acc);
            int32_t correction = bias * 128;

            C[(i+0) * N + j] = sum0 - correction;
            C[(i+1) * N + j] = sum1 - correction;
            C[(i+2) * N + j] = sum2 - correction;
            C[(i+3) * N + j] = sum3 - correction;
        }
    }
}

// ============================================================================
// Multi-threaded VNNI matmul with OpenMP
// Parallelizes the outer I loop
// ============================================================================
template<int N>
void vnni_matmul_multicore(const int8_t* A, const int8_t* B_T, int32_t* C) {
    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);

    // Parallelize the I loop (processing 4 rows at a time)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i += 4) {
        for (int j = 0; j < N; j++) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i bias_acc = _mm512_setzero_si512();

            for (int k = 0; k + 64 <= N; k += 64) {
                __m512i vb = _mm512_loadu_si512(&B_T[j * N + k]);

                __m512i va0 = _mm512_xor_si512(_mm512_loadu_si512(&A[(i+0) * N + k]), sign_flip);
                __m512i va1 = _mm512_xor_si512(_mm512_loadu_si512(&A[(i+1) * N + k]), sign_flip);
                __m512i va2 = _mm512_xor_si512(_mm512_loadu_si512(&A[(i+2) * N + k]), sign_flip);
                __m512i va3 = _mm512_xor_si512(_mm512_loadu_si512(&A[(i+3) * N + k]), sign_flip);

                acc0 = _mm512_dpbusd_epi32(acc0, va0, vb);
                acc1 = _mm512_dpbusd_epi32(acc1, va1, vb);
                acc2 = _mm512_dpbusd_epi32(acc2, va2, vb);
                acc3 = _mm512_dpbusd_epi32(acc3, va3, vb);

                bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, vb);
            }

            int32_t sum0 = _mm512_reduce_add_epi32(acc0);
            int32_t sum1 = _mm512_reduce_add_epi32(acc1);
            int32_t sum2 = _mm512_reduce_add_epi32(acc2);
            int32_t sum3 = _mm512_reduce_add_epi32(acc3);
            int32_t bias = _mm512_reduce_add_epi32(bias_acc);
            int32_t correction = bias * 128;

            C[(i+0) * N + j] = sum0 - correction;
            C[(i+1) * N + j] = sum1 - correction;
            C[(i+2) * N + j] = sum2 - correction;
            C[(i+3) * N + j] = sum3 - correction;
        }
    }
}

// ============================================================================
// Benchmark wrapper
// ============================================================================
template<int N>
struct BenchResult {
    double single_ms;
    double multi_ms;
    double single_giops;
    double multi_giops;
    double speedup;
    int32_t checksum;
    int num_threads;
};

template<int N>
BenchResult<N> benchmark_size(int iterations) {
    BenchResult<N> result = {};

    // Allocate aligned memory
    int8_t* A = (int8_t*)aligned_alloc(64, N * N * sizeof(int8_t));
    int8_t* B = (int8_t*)aligned_alloc(64, N * N * sizeof(int8_t));
    int8_t* B_T = (int8_t*)aligned_alloc(64, N * N * sizeof(int8_t));
    int32_t* C = (int32_t*)aligned_alloc(64, N * N * sizeof(int32_t));

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int val = ((i * N + j) % 127) - 64;
            A[i * N + j] = (int8_t)val;
            B[j * N + i] = (int8_t)val;
        }
    }

    // Transpose B
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            B_T[j * N + k] = B[k * N + j];
        }
    }

    result.num_threads = omp_get_max_threads();

    // Warmup
    for (int w = 0; w < 2; w++) {
        memset(C, 0, N * N * sizeof(int32_t));
        vnni_matmul_single_thread<N>(A, B_T, C);
    }

    // Benchmark single-threaded
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        memset(C, 0, N * N * sizeof(int32_t));
        vnni_matmul_single_thread<N>(A, B_T, C);
        DoNotOptimize(C[0]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    result.single_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Store checksum
    result.checksum = 0;
    for (int i = 0; i < N * N; i++) {
        result.checksum += C[i];
    }

    // Warmup multicore
    for (int w = 0; w < 2; w++) {
        memset(C, 0, N * N * sizeof(int32_t));
        vnni_matmul_multicore<N>(A, B_T, C);
    }

    // Benchmark multi-threaded
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        memset(C, 0, N * N * sizeof(int32_t));
        vnni_matmul_multicore<N>(A, B_T, C);
        DoNotOptimize(C[0]);
    }
    end = std::chrono::high_resolution_clock::now();
    result.multi_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Compute GIOP/s
    double giops = 2.0 * N * N * N / 1e9;
    result.single_giops = giops / (result.single_ms / 1000.0);
    result.multi_giops = giops / (result.multi_ms / 1000.0);
    result.speedup = result.single_ms / result.multi_ms;

    free(A);
    free(B);
    free(B_T);
    free(C);

    return result;
}

template<int N>
void print_result(const BenchResult<N>& r) {
    std::cout << std::setw(5) << N << "x" << std::setw(4) << N << " | ";
    std::cout << std::setw(10) << std::fixed << std::setprecision(3) << r.single_ms << " | ";
    std::cout << std::setw(9) << std::fixed << std::setprecision(2) << r.single_giops << " | ";
    std::cout << std::setw(10) << std::fixed << std::setprecision(3) << r.multi_ms << " | ";
    std::cout << std::setw(9) << std::fixed << std::setprecision(2) << r.multi_giops << " | ";
    std::cout << std::setw(6) << std::fixed << std::setprecision(2) << r.speedup << "x" << std::endl;
}

int main(int argc, char* argv[]) {
    int num_threads = omp_get_max_threads();

    std::cout << "=============================================================================" << std::endl;
    std::cout << "   INT8 VNNI MatMul Multicore Benchmark" << std::endl;
    std::cout << "=============================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "OpenMP threads: " << num_threads << std::endl;
    std::cout << std::endl;
    std::cout << "  Size   | Single (ms) |  GIOP/s   | Multi (ms)  |  GIOP/s   | Speedup" << std::endl;
    std::cout << "---------+-------------+-----------+-------------+-----------+---------" << std::endl;

    // Small sizes (more iterations for stable timing)
    print_result(benchmark_size<64>(50));
    print_result(benchmark_size<128>(20));
    print_result(benchmark_size<256>(10));
    print_result(benchmark_size<384>(5));
    print_result(benchmark_size<512>(5));
    print_result(benchmark_size<768>(3));
    print_result(benchmark_size<1024>(2));
    print_result(benchmark_size<2048>(1));

    std::cout << "=============================================================================" << std::endl;
    std::cout << std::endl;

    // Summary
    std::cout << "This benchmark shows the multicore ceiling achievable with OpenMP." << std::endl;
    std::cout << "SimpLang with OpenMP should achieve similar speedups." << std::endl;
    std::cout << std::endl;

    // Compare with TFLite ceiling (if known)
    std::cout << "Reference: TFLite XNNPACK multicore achieved ~1327 GIOP/s at 1024x1024" << std::endl;
    std::cout << "with " << num_threads << " threads." << std::endl;

    return 0;
}
