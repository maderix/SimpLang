/**
 * INT4 MatMul Benchmark Runner
 *
 * Compares SimpLang packed INT4 matmul against:
 * 1. Scalar reference (for correctness)
 * 2. Packed INT4 SIMD reference
 *
 * INT4 provides 2× memory savings vs INT8, ideal for:
 * - Weight-only quantization (W4)
 * - Extreme compression for edge deployment
 * - Cache-efficient inference
 *
 * Compile:
 *   g++ -O3 -march=native -mavx512bw bench_int4_matmul_runner.cpp -o bench_int4_matmul_runner -ldl
 */

#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cstring>
#include <immintrin.h>

typedef int32_t (*KernelFuncI32)();

// ============================================================================
// Reference Implementations
// ============================================================================

// Scalar reference implementation (for correctness verification)
// Signed INT4: values in range -8 to 7
template<int N>
int32_t scalar_matmul_int4() {
    // Store as int8_t but use only low 4 bits
    std::vector<int8_t> A(N * N);
    std::vector<int8_t> B(N * N);
    std::vector<int16_t> C(N * N, 0);

    // Initialize: INT4 range is -8 to 7 (4-bit signed)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Match SimpLang initialization: ((i * N + j) % 15) - 7
            int val = ((i * N + j) % 15) - 7;
            A[i * N + j] = (int8_t)val;
            B[j * N + i] = (int8_t)val;
        }
    }

    // Compute C = A × B (i4 × i4 → i16)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int16_t sum = 0;
            for (int k = 0; k < N; k++) {
                sum += (int16_t)A[i * N + k] * (int16_t)B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    // Compute checksum (as int32 to match SimpLang)
    int32_t checksum = 0;
    for (int i = 0; i < N * N; i++) {
        checksum += (int32_t)C[i];
    }
    return checksum;
}

// Packed INT4 reference - uses packed storage (2 values per byte)
// This matches EXACTLY what SimpLang does:
// - A is stored row-major, packed along columns (K dimension)
// - B is stored transposed (B_T), packed along K dimension (columns of B = rows of B_T)
template<int N>
int32_t packed_matmul_int4() {
    // Packed storage: N×(N/2) bytes
    const int packed_dim = (N + 1) / 2;
    std::vector<uint8_t> A_packed(N * packed_dim);
    std::vector<uint8_t> B_T_packed(N * packed_dim);  // B transposed, then packed along rows
    std::vector<int16_t> C(N * N, 0);

    // Helper to unpack and sign-extend i4 to i8
    auto unpack_i4 = [](uint8_t packed, int idx) -> int8_t {
        uint8_t nibble = (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
        // Sign extend from 4-bit
        if (nibble >= 8) nibble |= 0xF0;
        return (int8_t)nibble;
    };

    // Initialize A: packed along columns (K dimension)
    // A[i,j] = ((i * N + j) % 15) - 7
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 2) {
            int val0 = ((i * N + j) % 15) - 7;
            int val1 = (j + 1 < N) ? (((i * N + j + 1) % 15) - 7) : 0;

            // Pack: low nibble = val0 (even col), high nibble = val1 (odd col)
            uint8_t packed = ((uint8_t)(val0 & 0x0F)) | ((uint8_t)(val1 & 0x0F) << 4);
            A_packed[i * packed_dim + j/2] = packed;
        }
    }

    // Initialize B: In SimpLang, B[j,i] = ((i * N + j) % 15) - 7
    // This means B[row,col] = ((col * N + row) % 15) - 7
    // For matmul C[i,j] = sum_k A[i,k] * B[k,j]
    // We need B stored row-major, packed along columns (K dimension)
    // B[k,j] was stored as B[j,i] where j=k, i=j  => val = ((i * N + j) % 15) - 7 = ((j * N + k) % 15) - 7
    for (int k = 0; k < N; k++) {  // k = row of B
        for (int j = 0; j < N; j += 2) {  // j = column of B, packed in pairs
            // B[k,j] = ((j * N + k) % 15) - 7  (derived from transposed storage)
            int val0 = ((j * N + k) % 15) - 7;
            int val1 = (j + 1 < N) ? ((((j+1) * N + k) % 15) - 7) : 0;

            uint8_t packed = ((uint8_t)(val0 & 0x0F)) | ((uint8_t)(val1 & 0x0F) << 4);
            B_T_packed[k * packed_dim + j/2] = packed;
        }
    }

    // Compute C = A × B (unpacking on the fly)
    // C[i,j] = sum_k A[i,k] * B[k,j]
    // A is packed along K (columns), B is packed along J (columns)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int16_t sum = 0;
            for (int k = 0; k < N; k++) {
                // Unpack A[i,k]: row i, column k
                int a_byte = k / 2;
                int a_nibble = k % 2;
                int8_t a_val = unpack_i4(A_packed[i * packed_dim + a_byte], a_nibble);

                // Unpack B[k,j]: row k, column j
                int b_byte = j / 2;
                int b_nibble = j % 2;
                int8_t b_val = unpack_i4(B_T_packed[k * packed_dim + b_byte], b_nibble);

                sum += (int16_t)a_val * (int16_t)b_val;
            }
            C[i * N + j] = sum;
        }
    }

    // Compute checksum
    int32_t checksum = 0;
    for (int i = 0; i < N * N; i++) {
        checksum += (int32_t)C[i];
    }
    return checksum;
}

// SIMD-optimized packed INT4 matmul
// Uses AVX-512 to process multiple packed values
// Strategy: Transpose B to B_T to enable K-innermost vectorization
template<int N>
int32_t simd_matmul_int4() {
    const int packed_dim = (N + 1) / 2;
    alignas(64) uint8_t A_packed[N * packed_dim];
    alignas(64) uint8_t B_packed[N * packed_dim];      // B row-major, packed along J
    alignas(64) uint8_t B_T_packed[N * packed_dim];    // B transposed, packed along K
    alignas(64) int16_t C[N * N];
    memset(C, 0, sizeof(C));

    // Initialize and pack A: A[i,j] = ((i * N + j) % 15) - 7
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 2) {
            int val0 = ((i * N + j) % 15) - 7;
            int val1 = (j + 1 < N) ? (((i * N + j + 1) % 15) - 7) : 0;
            A_packed[i * packed_dim + j/2] = ((uint8_t)(val0 & 0x0F)) | ((uint8_t)(val1 & 0x0F) << 4);
        }
    }

    // Initialize B row-major: B[k,j] = ((j * N + k) % 15) - 7  (from transposed storage in SimpLang)
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j += 2) {
            int val0 = ((j * N + k) % 15) - 7;
            int val1 = (j + 1 < N) ? ((((j+1) * N + k) % 15) - 7) : 0;
            B_packed[k * packed_dim + j/2] = ((uint8_t)(val0 & 0x0F)) | ((uint8_t)(val1 & 0x0F) << 4);
        }
    }

    // Helper to unpack i4
    auto unpack_scalar = [](uint8_t packed, int idx) -> int8_t {
        uint8_t nibble = (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
        if (nibble >= 8) nibble |= 0xF0;
        return (int8_t)nibble;
    };

    // Transpose B to B_T for SIMD-efficient K-innermost access
    // B_T[j,k] = B[k,j]
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k += 2) {
            // Get B[k,j] and B[k+1,j]
            int8_t val0 = unpack_scalar(B_packed[k * packed_dim + j/2], j % 2);
            int8_t val1 = (k + 1 < N) ? unpack_scalar(B_packed[(k+1) * packed_dim + j/2], j % 2) : 0;
            // Pack into B_T[j, k:k+1]
            B_T_packed[j * packed_dim + k/2] = ((uint8_t)(val0 & 0x0F)) | ((uint8_t)(val1 & 0x0F) << 4);
        }
    }

    // SIMD matmul: C[i,j] = sum_k A[i,k] * B_T[j,k]
    // Both A and B_T are packed along K, enabling vectorized dot product

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m512i acc = _mm512_setzero_si512();

            // Process K dimension in chunks of 64 elements (32 packed bytes)
            int k = 0;
            for (; k + 63 < N; k += 64) {
                // Load 32 packed bytes (64 i4 values) from A row i
                __m256i a_packed_vec = _mm256_loadu_si256((__m256i*)&A_packed[i * packed_dim + k/2]);

                // Load 32 packed bytes from B_T row j
                __m256i b_packed_vec = _mm256_loadu_si256((__m256i*)&B_T_packed[j * packed_dim + k/2]);

                // Unpack to 512-bit (32 i16 values per register)
                __m512i a_full = _mm512_cvtepu8_epi16(a_packed_vec);
                __m512i b_full = _mm512_cvtepu8_epi16(b_packed_vec);

                // Extract low nibbles (even k indices)
                __m512i a_lo = _mm512_and_si512(a_full, _mm512_set1_epi16(0x000F));
                __m512i b_lo = _mm512_and_si512(b_full, _mm512_set1_epi16(0x000F));

                // Extract high nibbles (odd k indices)
                __m512i a_hi = _mm512_and_si512(_mm512_srli_epi16(a_full, 4), _mm512_set1_epi16(0x000F));
                __m512i b_hi = _mm512_and_si512(_mm512_srli_epi16(b_full, 4), _mm512_set1_epi16(0x000F));

                // Sign extend from i4 to i16
                __m512i sign_threshold = _mm512_set1_epi16(8);
                __m512i sixteen = _mm512_set1_epi16(16);

                __m512i a_lo_neg = _mm512_sub_epi16(a_lo, sixteen);
                __m512i a_hi_neg = _mm512_sub_epi16(a_hi, sixteen);
                __m512i b_lo_neg = _mm512_sub_epi16(b_lo, sixteen);
                __m512i b_hi_neg = _mm512_sub_epi16(b_hi, sixteen);

                a_lo = _mm512_mask_blend_epi16(_mm512_cmpge_epi16_mask(a_lo, sign_threshold), a_lo, a_lo_neg);
                a_hi = _mm512_mask_blend_epi16(_mm512_cmpge_epi16_mask(a_hi, sign_threshold), a_hi, a_hi_neg);
                b_lo = _mm512_mask_blend_epi16(_mm512_cmpge_epi16_mask(b_lo, sign_threshold), b_lo, b_lo_neg);
                b_hi = _mm512_mask_blend_epi16(_mm512_cmpge_epi16_mask(b_hi, sign_threshold), b_hi, b_hi_neg);

                // Multiply and accumulate
                __m512i prod_lo = _mm512_mullo_epi16(a_lo, b_lo);
                __m512i prod_hi = _mm512_mullo_epi16(a_hi, b_hi);

                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(prod_lo, _mm512_set1_epi16(1)));
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(prod_hi, _mm512_set1_epi16(1)));
            }

            int32_t sum = _mm512_reduce_add_epi32(acc);

            // Handle remaining elements with scalar code
            for (; k < N; k++) {
                int8_t a_val = unpack_scalar(A_packed[i * packed_dim + k/2], k % 2);
                int8_t b_val = unpack_scalar(B_T_packed[j * packed_dim + k/2], k % 2);
                sum += (int16_t)a_val * (int16_t)b_val;
            }

            C[i * N + j] = (int16_t)sum;
        }
    }

    // Compute checksum
    int32_t checksum = 0;
    for (int i = 0; i < N * N; i++) {
        checksum += (int32_t)C[i];
    }
    return checksum;
}

// ============================================================================
// Benchmarking Infrastructure
// ============================================================================

volatile int32_t g_sink = 0;

template<typename T>
inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template<typename Func>
double benchmark(Func func, int iterations) {
    // Warmup
    int32_t result;
    for (int w = 0; w < 5; w++) {
        result = func();
        DoNotOptimize(result);
    }

    // Benchmark
    int32_t accumulator = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        result = func();
        DoNotOptimize(result);
        accumulator ^= result;
    }
    auto end = std::chrono::high_resolution_clock::now();

    g_sink = accumulator;

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return elapsed_ms / iterations;
}

struct BenchmarkResult {
    int N;
    double simplang_ms;
    double packed_ms;
    double simd_ms;
    double simplang_giops;
    double packed_giops;
    double simd_giops;
    int32_t simplang_checksum;
    int32_t packed_checksum;
    int32_t simd_checksum;
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
    result.reference_checksum = scalar_matmul_int4<N>();

    // Benchmark SimpLang
    result.simplang_ms = benchmark(simplang_kernel, iterations);
    result.simplang_checksum = simplang_kernel();

    // Benchmark packed reference
    result.packed_ms = benchmark(packed_matmul_int4<N>, iterations);
    result.packed_checksum = packed_matmul_int4<N>();

    // Benchmark SIMD
    result.simd_ms = benchmark(simd_matmul_int4<N>, iterations);
    result.simd_checksum = simd_matmul_int4<N>();

    // Compute GIOP/s (2*N^3 operations for matmul)
    double giops = 2.0 * N * N * N / 1e9;
    result.simplang_giops = giops / (result.simplang_ms / 1000.0);
    result.packed_giops = giops / (result.packed_ms / 1000.0);
    result.simd_giops = giops / (result.simd_ms / 1000.0);

    // Check correctness - allow small differences due to accumulation order
    result.correct = (result.simplang_checksum == result.reference_checksum) &&
                     (result.packed_checksum == result.reference_checksum);

    return result;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::setw(5) << r.N << "×" << std::setw(4) << r.N << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << r.simplang_ms << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << r.simplang_giops << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << r.packed_ms << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << r.packed_giops << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << r.simd_ms << " │ ";
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << r.simd_giops << " │ ";

    // Ratio vs SIMD
    double ratio_vs_simd = (r.simd_ms / r.simplang_ms) * 100.0;
    std::cout << std::setw(7) << std::fixed << std::setprecision(1) << ratio_vs_simd << "% │";

    if (r.correct) {
        std::cout << " ✓" << std::endl;
    } else {
        std::cout << " ✗" << std::endl;
        std::cout << "       Checksums: SL=" << r.simplang_checksum
                  << " Packed=" << r.packed_checksum
                  << " SIMD=" << r.simd_checksum
                  << " Ref=" << r.reference_checksum << std::endl;
    }
}

int main(int argc, char* argv[]) {
    const char* so_path = "/tmp/bench_int4.so";
    if (argc > 1) {
        so_path = argv[1];
    }

    std::cout << "═══════════════════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   INT4 GEMM Benchmark: SimpLang Packed INT4 vs Reference" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << "INT4 provides 2× memory savings vs INT8 with packed storage (2 values per byte)" << std::endl;
    std::cout << "Loading: " << so_path << std::endl;

    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << " Size   │ SimpLang │  GIOP/s  │  Packed  │  GIOP/s  │   SIMD   │  GIOP/s  │ vs SIMD │ OK" << std::endl;
    std::cout << "────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼─────────┼────" << std::endl;

    // Run benchmarks
    print_result(benchmark_size<32>(handle, "benchmark_int4_matmul_32", 100));
    print_result(benchmark_size<64>(handle, "benchmark_int4_matmul_64", 50));
    print_result(benchmark_size<128>(handle, "benchmark_int4_matmul_128", 20));
    print_result(benchmark_size<256>(handle, "benchmark_int4_matmul_256", 10));
    print_result(benchmark_size<384>(handle, "benchmark_int4_matmul_384", 5));
    print_result(benchmark_size<512>(handle, "benchmark_int4_matmul_512", 3));
    print_result(benchmark_size<768>(handle, "benchmark_int4_matmul_768", 2));
    print_result(benchmark_size<1024>(handle, "benchmark_int4_matmul_1024", 1));

    std::cout << "═══════════════════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    std::cout << "Legend:" << std::endl;
    std::cout << "  • GIOP/s = Giga Integer Operations per second (higher is better)" << std::endl;
    std::cout << "  • vs SIMD = (SIMD time / SimpLang time) × 100%" << std::endl;
    std::cout << "  • >100% means SimpLang is faster than SIMD reference" << std::endl;
    std::cout << "  • <100% means SimpLang is slower than SIMD reference" << std::endl;
    std::cout << std::endl;

    dlclose(handle);
    return 0;
}
