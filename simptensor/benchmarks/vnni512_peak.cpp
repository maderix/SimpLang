/**
 * AVX512-VNNI Peak Performance Benchmark
 *
 * Tests true VNNI throughput using vpdpbusd instruction:
 * - vpdpbusd: 16 x (4 × u8×i8 → i32) per instruction = 128 int8 ops
 *
 * Compile:
 *   g++ -O3 -march=native -mavx512vnni vnni512_peak.cpp -o vnni512_peak
 */

#include <iostream>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <iomanip>

// Prevent compiler from optimizing away
volatile int32_t g_sink = 0;

template<typename T>
inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

//=============================================================================
// Test 1: Pure VNNI throughput (no memory, just registers)
//=============================================================================
int64_t vnni_register_throughput(int64_t iterations) {
    __m512i a = _mm512_set1_epi8(1);
    __m512i b = _mm512_set1_epi8(1);
    __m512i c0 = _mm512_setzero_si512();
    __m512i c1 = _mm512_setzero_si512();
    __m512i c2 = _mm512_setzero_si512();
    __m512i c3 = _mm512_setzero_si512();
    __m512i c4 = _mm512_setzero_si512();
    __m512i c5 = _mm512_setzero_si512();
    __m512i c6 = _mm512_setzero_si512();
    __m512i c7 = _mm512_setzero_si512();

    // 8 independent chains to saturate execution units
    for (int64_t i = 0; i < iterations; i++) {
        // Each vpdpbusd: 16 lanes × 4 u8×i8 mults = 64 mults + 64 adds = 128 ops
        c0 = _mm512_dpbusd_epi32(c0, a, b);
        c1 = _mm512_dpbusd_epi32(c1, a, b);
        c2 = _mm512_dpbusd_epi32(c2, a, b);
        c3 = _mm512_dpbusd_epi32(c3, a, b);
        c4 = _mm512_dpbusd_epi32(c4, a, b);
        c5 = _mm512_dpbusd_epi32(c5, a, b);
        c6 = _mm512_dpbusd_epi32(c6, a, b);
        c7 = _mm512_dpbusd_epi32(c7, a, b);
    }

    // Prevent optimization
    __m512i sum = _mm512_add_epi32(c0, c1);
    sum = _mm512_add_epi32(sum, c2);
    sum = _mm512_add_epi32(sum, c3);
    sum = _mm512_add_epi32(sum, c4);
    sum = _mm512_add_epi32(sum, c5);
    sum = _mm512_add_epi32(sum, c6);
    sum = _mm512_add_epi32(sum, c7);

    return _mm512_reduce_add_epi32(sum);
}

//=============================================================================
// Test 2: VNNI with L1 cache access pattern
//=============================================================================
int64_t vnni_l1_throughput(int64_t iterations) {
    // Small buffers that fit in L1 (32KB)
    alignas(64) int8_t A[4096];  // 4KB
    alignas(64) int8_t B[4096];  // 4KB
    alignas(64) int32_t C[1024]; // 4KB

    memset(A, 1, sizeof(A));
    memset(B, 1, sizeof(B));
    memset(C, 0, sizeof(C));

    for (int64_t iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 4096; i += 64) {
            __m512i a = _mm512_load_si512((__m512i*)&A[i]);
            __m512i b = _mm512_load_si512((__m512i*)&B[i]);
            __m512i c = _mm512_load_si512((__m512i*)&C[i/4]);
            c = _mm512_dpbusd_epi32(c, a, b);
            _mm512_store_si512((__m512i*)&C[i/4], c);
        }
    }

    int64_t sum = 0;
    for (int i = 0; i < 1024; i++) sum += C[i];
    return sum;
}

//=============================================================================
// Test 3: VNNI MatMul micro-kernel (8x8 output tile)
//=============================================================================
void vnni_matmul_8x8(const int8_t* A, const int8_t* B, int32_t* C,
                     int M, int N, int K) {
    // A: M×K (row-major)
    // B: K×N (row-major)
    // C: M×N (row-major)

    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 16) {
            // 8 accumulators for 8 rows
            __m512i c0 = _mm512_setzero_si512();
            __m512i c1 = _mm512_setzero_si512();
            __m512i c2 = _mm512_setzero_si512();
            __m512i c3 = _mm512_setzero_si512();
            __m512i c4 = _mm512_setzero_si512();
            __m512i c5 = _mm512_setzero_si512();
            __m512i c6 = _mm512_setzero_si512();
            __m512i c7 = _mm512_setzero_si512();

            // K must be multiple of 4 for vpdpbusd
            for (int k = 0; k < K; k += 4) {
                // Broadcast 4 elements from each A row
                __m512i a0 = _mm512_set1_epi32(*(int32_t*)&A[(i+0)*K + k]);
                __m512i a1 = _mm512_set1_epi32(*(int32_t*)&A[(i+1)*K + k]);
                __m512i a2 = _mm512_set1_epi32(*(int32_t*)&A[(i+2)*K + k]);
                __m512i a3 = _mm512_set1_epi32(*(int32_t*)&A[(i+3)*K + k]);
                __m512i a4 = _mm512_set1_epi32(*(int32_t*)&A[(i+4)*K + k]);
                __m512i a5 = _mm512_set1_epi32(*(int32_t*)&A[(i+5)*K + k]);
                __m512i a6 = _mm512_set1_epi32(*(int32_t*)&A[(i+6)*K + k]);
                __m512i a7 = _mm512_set1_epi32(*(int32_t*)&A[(i+7)*K + k]);

                // Load B row (16 columns × 4 k-elements packed)
                // B needs to be in VNNI layout: K/4 × N × 4
                __m512i b = _mm512_loadu_si512((__m512i*)&B[k*N + j*4]);

                c0 = _mm512_dpbusd_epi32(c0, a0, b);
                c1 = _mm512_dpbusd_epi32(c1, a1, b);
                c2 = _mm512_dpbusd_epi32(c2, a2, b);
                c3 = _mm512_dpbusd_epi32(c3, a3, b);
                c4 = _mm512_dpbusd_epi32(c4, a4, b);
                c5 = _mm512_dpbusd_epi32(c5, a5, b);
                c6 = _mm512_dpbusd_epi32(c6, a6, b);
                c7 = _mm512_dpbusd_epi32(c7, a7, b);
            }

            // Store results
            _mm512_storeu_si512((__m512i*)&C[(i+0)*N + j], c0);
            _mm512_storeu_si512((__m512i*)&C[(i+1)*N + j], c1);
            _mm512_storeu_si512((__m512i*)&C[(i+2)*N + j], c2);
            _mm512_storeu_si512((__m512i*)&C[(i+3)*N + j], c3);
            _mm512_storeu_si512((__m512i*)&C[(i+4)*N + j], c4);
            _mm512_storeu_si512((__m512i*)&C[(i+5)*N + j], c5);
            _mm512_storeu_si512((__m512i*)&C[(i+6)*N + j], c6);
            _mm512_storeu_si512((__m512i*)&C[(i+7)*N + j], c7);
        }
    }
}

//=============================================================================
// Test 4: Full VNNI MatMul with proper data layout
//=============================================================================
template<int N>
int64_t vnni_matmul_benchmark() {
    // Allocate aligned buffers
    alignas(64) int8_t A[N * N];
    alignas(64) int8_t B_vnni[N * N];  // VNNI layout: (K/4) × N × 4
    alignas(64) int32_t C[N * N];

    // Initialize A (row-major)
    for (int i = 0; i < N * N; i++) {
        A[i] = (i % 127) - 64;
    }

    // Initialize B in VNNI-friendly layout
    // Original B[k,j], VNNI layout: B_vnni[k/4][j][k%4]
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            int8_t val = ((k * N + j) % 127) - 64;
            // Pack 4 k-values together for each j
            B_vnni[(k/4) * N * 4 + j * 4 + (k % 4)] = val;
        }
    }

    memset(C, 0, sizeof(C));

    // Warmup
    vnni_matmul_8x8(A, B_vnni, C, N, N, N);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    vnni_matmul_8x8(A, B_vnni, C, N, N, N);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gops = 2.0 * N * N * N / 1e9;
    double giops = gops / (ms / 1000.0);

    std::cout << std::setw(5) << N << "×" << std::setw(4) << N
              << " │ " << std::setw(10) << std::fixed << std::setprecision(3) << ms << " ms"
              << " │ " << std::setw(10) << std::fixed << std::setprecision(2) << giops << " GIOP/s"
              << std::endl;

    int64_t checksum = 0;
    for (int i = 0; i < N * N; i++) checksum += C[i];
    return checksum;
}

//=============================================================================
// Main
//=============================================================================
int main() {
    std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   AVX512-VNNI Peak Performance Benchmark (vpdpbusd)" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;

    // Test 1: Pure register throughput
    std::cout << "=== Test 1: Pure VNNI Register Throughput ===" << std::endl;
    {
        int64_t iterations = 100000000;  // 100M iterations

        auto start = std::chrono::high_resolution_clock::now();
        int64_t result = vnni_register_throughput(iterations);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        // 8 vpdpbusd per iteration, each does 128 int8 ops
        double total_ops = (double)iterations * 8 * 128;
        double giops = total_ops / 1e9 / (ms / 1000.0);

        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Time: " << std::fixed << std::setprecision(2) << ms << " ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) << giops << " GIOP/s" << std::endl;
        std::cout << "Result (prevent opt): " << result << std::endl;
        std::cout << std::endl;
    }

    // Test 2: L1 cache throughput
    std::cout << "=== Test 2: VNNI with L1 Cache ===" << std::endl;
    {
        int64_t iterations = 1000000;

        auto start = std::chrono::high_resolution_clock::now();
        int64_t result = vnni_l1_throughput(iterations);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        // 4096/64 = 64 vpdpbusd per iteration, each does 128 ops
        double total_ops = (double)iterations * 64 * 128;
        double giops = total_ops / 1e9 / (ms / 1000.0);

        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Time: " << std::fixed << std::setprecision(2) << ms << " ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) << giops << " GIOP/s" << std::endl;
        std::cout << "Result (prevent opt): " << result << std::endl;
        std::cout << std::endl;
    }

    // Test 3: MatMul with VNNI
    std::cout << "=== Test 3: VNNI MatMul (8×8 micro-kernel) ===" << std::endl;
    std::cout << " Size   │    Time      │  Throughput" << std::endl;
    std::cout << "────────┼──────────────┼─────────────" << std::endl;

    vnni_matmul_benchmark<64>();
    vnni_matmul_benchmark<128>();
    vnni_matmul_benchmark<256>();
    vnni_matmul_benchmark<512>();
    vnni_matmul_benchmark<1024>();

    std::cout << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════" << std::endl;

    return 0;
}
