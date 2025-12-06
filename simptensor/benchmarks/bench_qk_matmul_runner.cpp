/**
 * QK^T MatMul Benchmark Runner for LLaMA 3.2 1B Self-Attention
 *
 * Tests INT8 QK^T computation:
 *   Q: [seq_len, head_dim] = [1024, 64]
 *   K^T: [head_dim, seq_len] = [64, 1024]
 *   scores: [seq_len, seq_len] = [1024, 1024]
 *
 * Compile:
 *   g++ -O3 -march=native -mavx512vnni -o bench_qk_matmul_runner \
 *       bench_qk_matmul_runner.cpp -ldl
 */

#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cstring>
#include <immintrin.h>

typedef int32_t (*KernelFunc)();

// ============================================================================
// Constants
// ============================================================================
constexpr int SEQ_LEN = 1024;
constexpr int HEAD_DIM = 64;

// ============================================================================
// Reference Implementations
// ============================================================================

// Scalar reference for single head
int32_t scalar_qk_matmul_1head() {
    std::vector<int8_t> Q(SEQ_LEN * HEAD_DIM);
    std::vector<int8_t> K_T(HEAD_DIM * SEQ_LEN);
    std::vector<int32_t> scores(SEQ_LEN * SEQ_LEN, 0);

    // Initialize
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            int val = ((i * HEAD_DIM + j) % 127) - 64;
            Q[i * HEAD_DIM + j] = (int8_t)val;
            K_T[j * SEQ_LEN + i] = (int8_t)val;
        }
    }

    // QK^T
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            int32_t sum = 0;
            for (int k = 0; k < HEAD_DIM; k++) {
                sum += (int32_t)Q[i * HEAD_DIM + k] * (int32_t)K_T[k * SEQ_LEN + j];
            }
            scores[i * SEQ_LEN + j] = sum;
        }
    }

    int32_t checksum = 0;
    for (int i = 0; i < SEQ_LEN * SEQ_LEN; i++) {
        checksum += scores[i];
    }
    return checksum;
}

// AVX512-VNNI optimized single head with I=4 tiling
int32_t vnni_qk_matmul_1head() {
    int8_t* Q = (int8_t*)aligned_alloc(64, SEQ_LEN * HEAD_DIM);
    int8_t* K_T = (int8_t*)aligned_alloc(64, HEAD_DIM * SEQ_LEN);
    int32_t* scores = (int32_t*)aligned_alloc(64, SEQ_LEN * SEQ_LEN * sizeof(int32_t));
    memset(scores, 0, SEQ_LEN * SEQ_LEN * sizeof(int32_t));

    // Initialize
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            int val = ((i * HEAD_DIM + j) % 127) - 64;
            Q[i * HEAD_DIM + j] = (int8_t)val;
            K_T[j * SEQ_LEN + i] = (int8_t)val;
        }
    }

    // Transpose K_T to K for SIMD-friendly access
    // K[j][k] = K_T[k][j], so K is [SEQ_LEN, HEAD_DIM]
    int8_t* K = (int8_t*)aligned_alloc(64, SEQ_LEN * HEAD_DIM);
    for (int j = 0; j < SEQ_LEN; j++) {
        for (int k = 0; k < HEAD_DIM; k++) {
            K[j * HEAD_DIM + k] = K_T[k * SEQ_LEN + j];
        }
    }

    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);

    // I=4 tiled matmul using vpdpbusd
    // HEAD_DIM=64 = exactly one 512-bit vector
    for (int i = 0; i < SEQ_LEN; i += 4) {
        for (int j = 0; j < SEQ_LEN; j++) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i bias_acc = _mm512_setzero_si512();

            // Load K[j] once (64 bytes = HEAD_DIM)
            __m512i vk = _mm512_loadu_si512(&K[j * HEAD_DIM]);

            // Load 4 Q rows and XOR with 0x80
            __m512i vq0 = _mm512_xor_si512(_mm512_loadu_si512(&Q[(i+0) * HEAD_DIM]), sign_flip);
            __m512i vq1 = _mm512_xor_si512(_mm512_loadu_si512(&Q[(i+1) * HEAD_DIM]), sign_flip);
            __m512i vq2 = _mm512_xor_si512(_mm512_loadu_si512(&Q[(i+2) * HEAD_DIM]), sign_flip);
            __m512i vq3 = _mm512_xor_si512(_mm512_loadu_si512(&Q[(i+3) * HEAD_DIM]), sign_flip);

            // vpdpbusd
            acc0 = _mm512_dpbusd_epi32(acc0, vq0, vk);
            acc1 = _mm512_dpbusd_epi32(acc1, vq1, vk);
            acc2 = _mm512_dpbusd_epi32(acc2, vq2, vk);
            acc3 = _mm512_dpbusd_epi32(acc3, vq3, vk);
            bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, vk);

            // Horizontal reduction
            int32_t sum0 = _mm512_reduce_add_epi32(acc0);
            int32_t sum1 = _mm512_reduce_add_epi32(acc1);
            int32_t sum2 = _mm512_reduce_add_epi32(acc2);
            int32_t sum3 = _mm512_reduce_add_epi32(acc3);
            int32_t bias = _mm512_reduce_add_epi32(bias_acc);
            int32_t correction = bias * 128;

            scores[(i+0) * SEQ_LEN + j] = sum0 - correction;
            scores[(i+1) * SEQ_LEN + j] = sum1 - correction;
            scores[(i+2) * SEQ_LEN + j] = sum2 - correction;
            scores[(i+3) * SEQ_LEN + j] = sum3 - correction;
        }
    }

    int32_t checksum = 0;
    for (int i = 0; i < SEQ_LEN * SEQ_LEN; i++) {
        checksum += scores[i];
    }

    free(Q);
    free(K_T);
    free(K);
    free(scores);
    return checksum;
}

// Multi-head versions
template<int NUM_HEADS>
int32_t scalar_qk_matmul_multihead() {
    std::vector<int8_t> Q(NUM_HEADS * SEQ_LEN * HEAD_DIM);
    std::vector<int8_t> K_T(NUM_HEADS * HEAD_DIM * SEQ_LEN);
    std::vector<int32_t> scores(NUM_HEADS * SEQ_LEN * SEQ_LEN, 0);

    // Initialize
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                int idx = h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + j;
                int val = (idx % 127) - 64;
                Q[idx] = (int8_t)val;
                K_T[h * HEAD_DIM * SEQ_LEN + j * SEQ_LEN + i] = (int8_t)val;
            }
        }
    }

    // Batched QK^T
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < SEQ_LEN; j++) {
                int32_t sum = 0;
                for (int k = 0; k < HEAD_DIM; k++) {
                    int q_idx = h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + k;
                    int k_idx = h * HEAD_DIM * SEQ_LEN + k * SEQ_LEN + j;
                    sum += (int32_t)Q[q_idx] * (int32_t)K_T[k_idx];
                }
                scores[h * SEQ_LEN * SEQ_LEN + i * SEQ_LEN + j] = sum;
            }
        }
    }

    int32_t checksum = 0;
    for (size_t i = 0; i < scores.size(); i++) {
        checksum += scores[i];
    }
    return checksum;
}

template<int NUM_HEADS>
int32_t vnni_qk_matmul_multihead() {
    int8_t* Q = (int8_t*)aligned_alloc(64, NUM_HEADS * SEQ_LEN * HEAD_DIM);
    int8_t* K_T = (int8_t*)aligned_alloc(64, NUM_HEADS * HEAD_DIM * SEQ_LEN);
    int8_t* K = (int8_t*)aligned_alloc(64, NUM_HEADS * SEQ_LEN * HEAD_DIM);
    int32_t* scores = (int32_t*)aligned_alloc(64, NUM_HEADS * SEQ_LEN * SEQ_LEN * sizeof(int32_t));
    memset(scores, 0, NUM_HEADS * SEQ_LEN * SEQ_LEN * sizeof(int32_t));

    // Initialize
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                int idx = h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + j;
                int val = (idx % 127) - 64;
                Q[idx] = (int8_t)val;
                K_T[h * HEAD_DIM * SEQ_LEN + j * SEQ_LEN + i] = (int8_t)val;
            }
        }
    }

    // Transpose K_T to K for each head
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            for (int k = 0; k < HEAD_DIM; k++) {
                K[h * SEQ_LEN * HEAD_DIM + j * HEAD_DIM + k] =
                    K_T[h * HEAD_DIM * SEQ_LEN + k * SEQ_LEN + j];
            }
        }
    }

    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);

    // Batched I=4 tiled matmul
    for (int h = 0; h < NUM_HEADS; h++) {
        int8_t* Q_h = Q + h * SEQ_LEN * HEAD_DIM;
        int8_t* K_h = K + h * SEQ_LEN * HEAD_DIM;
        int32_t* scores_h = scores + h * SEQ_LEN * SEQ_LEN;

        for (int i = 0; i < SEQ_LEN; i += 4) {
            for (int j = 0; j < SEQ_LEN; j++) {
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                __m512i bias_acc = _mm512_setzero_si512();

                __m512i vk = _mm512_loadu_si512(&K_h[j * HEAD_DIM]);

                __m512i vq0 = _mm512_xor_si512(_mm512_loadu_si512(&Q_h[(i+0) * HEAD_DIM]), sign_flip);
                __m512i vq1 = _mm512_xor_si512(_mm512_loadu_si512(&Q_h[(i+1) * HEAD_DIM]), sign_flip);
                __m512i vq2 = _mm512_xor_si512(_mm512_loadu_si512(&Q_h[(i+2) * HEAD_DIM]), sign_flip);
                __m512i vq3 = _mm512_xor_si512(_mm512_loadu_si512(&Q_h[(i+3) * HEAD_DIM]), sign_flip);

                acc0 = _mm512_dpbusd_epi32(acc0, vq0, vk);
                acc1 = _mm512_dpbusd_epi32(acc1, vq1, vk);
                acc2 = _mm512_dpbusd_epi32(acc2, vq2, vk);
                acc3 = _mm512_dpbusd_epi32(acc3, vq3, vk);
                bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, vk);

                int32_t sum0 = _mm512_reduce_add_epi32(acc0);
                int32_t sum1 = _mm512_reduce_add_epi32(acc1);
                int32_t sum2 = _mm512_reduce_add_epi32(acc2);
                int32_t sum3 = _mm512_reduce_add_epi32(acc3);
                int32_t bias = _mm512_reduce_add_epi32(bias_acc);
                int32_t correction = bias * 128;

                scores_h[(i+0) * SEQ_LEN + j] = sum0 - correction;
                scores_h[(i+1) * SEQ_LEN + j] = sum1 - correction;
                scores_h[(i+2) * SEQ_LEN + j] = sum2 - correction;
                scores_h[(i+3) * SEQ_LEN + j] = sum3 - correction;
            }
        }
    }

    int32_t checksum = 0;
    for (int i = 0; i < NUM_HEADS * SEQ_LEN * SEQ_LEN; i++) {
        checksum += scores[i];
    }

    free(Q);
    free(K_T);
    free(K);
    free(scores);
    return checksum;
}

// ============================================================================
// Benchmarking
// ============================================================================
volatile int32_t g_sink = 0;

template<typename T>
inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template<typename Func>
double benchmark(Func func, int iterations) {
    int32_t result;
    for (int w = 0; w < 3; w++) {
        result = func();
        DoNotOptimize(result);
    }

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

int main(int argc, char* argv[]) {
    const char* so_path = "/tmp/bench_qk_matmul.so";
    if (argc > 1) so_path = argv[1];

    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "   QK^T MatMul Benchmark (LLaMA 3.2 1B Self-Attention)" << std::endl;
    std::cout << "   SEQ_LEN=" << SEQ_LEN << ", HEAD_DIM=" << HEAD_DIM << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════════" << std::endl;

    void* handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load " << so_path << ": " << dlerror() << std::endl;
        std::cerr << "Running reference benchmarks only..." << std::endl;
    }

    // Reference checksums
    int32_t ref_1head = scalar_qk_matmul_1head();
    int32_t ref_4head = scalar_qk_matmul_multihead<4>();
    int32_t ref_32head = scalar_qk_matmul_multihead<32>();

    std::cout << "\n=== Single Head (1024x64 × 64x1024 → 1024x1024) ===" << std::endl;
    std::cout << "Reference checksum: " << ref_1head << std::endl;

    double scalar_1head_ms = benchmark(scalar_qk_matmul_1head, 10);
    double vnni_1head_ms = benchmark(vnni_qk_matmul_1head, 10);

    // GIOP/s: 2 * SEQ_LEN * SEQ_LEN * HEAD_DIM ops
    double giops_1head = 2.0 * SEQ_LEN * SEQ_LEN * HEAD_DIM / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Scalar:    " << scalar_1head_ms << " ms ("
              << std::setprecision(2) << giops_1head / (scalar_1head_ms / 1000.0) << " GIOP/s)" << std::endl;
    std::cout << "VNNI C++:  " << std::setprecision(3) << vnni_1head_ms << " ms ("
              << std::setprecision(2) << giops_1head / (vnni_1head_ms / 1000.0) << " GIOP/s)" << std::endl;

    int32_t vnni_check = vnni_qk_matmul_1head();
    std::cout << "VNNI checksum: " << vnni_check << (vnni_check == ref_1head ? " ✓" : " ✗") << std::endl;

    if (handle) {
        KernelFunc sl_1head = (KernelFunc)dlsym(handle, "benchmark_qk_matmul_1head");
        if (sl_1head) {
            double sl_1head_ms = benchmark(sl_1head, 10);
            int32_t sl_check = sl_1head();
            std::cout << "SimpLang:  " << std::setprecision(3) << sl_1head_ms << " ms ("
                      << std::setprecision(2) << giops_1head / (sl_1head_ms / 1000.0) << " GIOP/s) "
                      << std::setprecision(1) << (vnni_1head_ms / sl_1head_ms * 100.0) << "% vs VNNI"
                      << std::endl;
            std::cout << "SimpLang checksum: " << sl_check << (sl_check == ref_1head ? " ✓" : " (mismatch, debug needed)") << std::endl;
        }
    }

    std::cout << "\n=== 4 Heads ===" << std::endl;
    std::cout << "Reference checksum: " << ref_4head << std::endl;

    double scalar_4head_ms = benchmark(scalar_qk_matmul_multihead<4>, 5);
    double vnni_4head_ms = benchmark(vnni_qk_matmul_multihead<4>, 5);
    double giops_4head = 4.0 * 2.0 * SEQ_LEN * SEQ_LEN * HEAD_DIM / 1e9;

    std::cout << "Scalar:    " << std::setprecision(3) << scalar_4head_ms << " ms ("
              << std::setprecision(2) << giops_4head / (scalar_4head_ms / 1000.0) << " GIOP/s)" << std::endl;
    std::cout << "VNNI C++:  " << std::setprecision(3) << vnni_4head_ms << " ms ("
              << std::setprecision(2) << giops_4head / (vnni_4head_ms / 1000.0) << " GIOP/s)" << std::endl;

    if (handle) {
        KernelFunc sl_4head = (KernelFunc)dlsym(handle, "benchmark_qk_matmul_4head");
        if (sl_4head) {
            double sl_4head_ms = benchmark(sl_4head, 5);
            int32_t sl_check = sl_4head();
            std::cout << "SimpLang:  " << std::setprecision(3) << sl_4head_ms << " ms ("
                      << std::setprecision(2) << giops_4head / (sl_4head_ms / 1000.0) << " GIOP/s) "
                      << std::setprecision(1) << (vnni_4head_ms / sl_4head_ms * 100.0) << "% vs VNNI"
                      << (sl_check == ref_4head ? " ✓" : " ✗") << std::endl;
        }
    }

    std::cout << "\n=== 32 Heads (Full LLaMA 3.2 1B) ===" << std::endl;
    std::cout << "Reference checksum: " << ref_32head << std::endl;

    double scalar_32head_ms = benchmark(scalar_qk_matmul_multihead<32>, 1);
    double vnni_32head_ms = benchmark(vnni_qk_matmul_multihead<32>, 1);
    double giops_32head = 32.0 * 2.0 * SEQ_LEN * SEQ_LEN * HEAD_DIM / 1e9;

    std::cout << "Scalar:    " << std::setprecision(3) << scalar_32head_ms << " ms ("
              << std::setprecision(2) << giops_32head / (scalar_32head_ms / 1000.0) << " GIOP/s)" << std::endl;
    std::cout << "VNNI C++:  " << std::setprecision(3) << vnni_32head_ms << " ms ("
              << std::setprecision(2) << giops_32head / (vnni_32head_ms / 1000.0) << " GIOP/s)" << std::endl;

    if (handle) {
        KernelFunc sl_32head = (KernelFunc)dlsym(handle, "benchmark_qk_matmul_32head");
        if (sl_32head) {
            double sl_32head_ms = benchmark(sl_32head, 1);
            int32_t sl_check = sl_32head();
            std::cout << "SimpLang:  " << std::setprecision(3) << sl_32head_ms << " ms ("
                      << std::setprecision(2) << giops_32head / (sl_32head_ms / 1000.0) << " GIOP/s) "
                      << std::setprecision(1) << (vnni_32head_ms / sl_32head_ms * 100.0) << "% vs VNNI"
                      << (sl_check == ref_32head ? " ✓" : " ✗") << std::endl;
        }
    }

    std::cout << "\n═══════════════════════════════════════════════════════════════════════════════" << std::endl;

    if (handle) dlclose(handle);
    return 0;
}
