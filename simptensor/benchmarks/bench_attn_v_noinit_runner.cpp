/**
 * Attention × V MatMul Benchmark Runner - NO INIT version
 * Init done outside measurement, only matmul timed
 */

#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cstring>
#include <immintrin.h>

constexpr int SEQ_LEN = 1024;
constexpr int HEAD_DIM = 64;
constexpr int WARMUP = 3;
constexpr int ITERATIONS = 10;

// MLIR MemRef format: base_ptr, aligned_ptr, offset, size, stride
#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_I8(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

// SimpLang function signatures with MLIR memref ABI
typedef int32_t (*Kernel1Head)(MEMREF_I8_PARAMS, MEMREF_I8_PARAMS);
typedef int32_t (*Kernel4Head)(
    MEMREF_I8_PARAMS, MEMREF_I8_PARAMS,
    MEMREF_I8_PARAMS, MEMREF_I8_PARAMS,
    MEMREF_I8_PARAMS, MEMREF_I8_PARAMS,
    MEMREF_I8_PARAMS, MEMREF_I8_PARAMS
);

// Pre-allocated global buffers (init once)
alignas(64) int8_t g_Attn[4][SEQ_LEN * SEQ_LEN];
alignas(64) int8_t g_V[4][SEQ_LEN * HEAD_DIM];
alignas(64) int8_t g_V_T[4][HEAD_DIM * SEQ_LEN];  // Transposed for C++ VNNI
alignas(64) int32_t g_output[SEQ_LEN * HEAD_DIM];

void init_data() {
    for (int h = 0; h < 4; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < SEQ_LEN; j++) {
                int idx = h * SEQ_LEN * SEQ_LEN + i * SEQ_LEN + j;
                g_Attn[h][i * SEQ_LEN + j] = (int8_t)((idx % 127) - 64);
            }
        }
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                int idx = h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + j;
                g_V[h][i * HEAD_DIM + j] = (int8_t)((idx % 127) - 64);
            }
        }
        // Transpose V for C++ VNNI
        for (int j = 0; j < HEAD_DIM; j++) {
            for (int k = 0; k < SEQ_LEN; k++) {
                g_V_T[h][j * SEQ_LEN + k] = g_V[h][k * HEAD_DIM + j];
            }
        }
    }
}

// C++ VNNI single head - uses pre-transposed V_T
int32_t vnni_1head_noinit(int8_t* Attn, int8_t* V_T) {
    memset(g_output, 0, sizeof(g_output));

    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);

    for (int i = 0; i < SEQ_LEN; i += 4) {
        for (int j = 0; j < HEAD_DIM; j++) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i bias_acc = _mm512_setzero_si512();

            for (int k = 0; k < SEQ_LEN; k += 64) {
                __m512i vv = _mm512_loadu_si512(&V_T[j * SEQ_LEN + k]);
                __m512i va0 = _mm512_xor_si512(_mm512_loadu_si512(&Attn[(i+0) * SEQ_LEN + k]), sign_flip);
                __m512i va1 = _mm512_xor_si512(_mm512_loadu_si512(&Attn[(i+1) * SEQ_LEN + k]), sign_flip);
                __m512i va2 = _mm512_xor_si512(_mm512_loadu_si512(&Attn[(i+2) * SEQ_LEN + k]), sign_flip);
                __m512i va3 = _mm512_xor_si512(_mm512_loadu_si512(&Attn[(i+3) * SEQ_LEN + k]), sign_flip);

                acc0 = _mm512_dpbusd_epi32(acc0, va0, vv);
                acc1 = _mm512_dpbusd_epi32(acc1, va1, vv);
                acc2 = _mm512_dpbusd_epi32(acc2, va2, vv);
                acc3 = _mm512_dpbusd_epi32(acc3, va3, vv);
                bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, vv);
            }

            int32_t bias = _mm512_reduce_add_epi32(bias_acc) * 128;
            g_output[(i+0) * HEAD_DIM + j] = _mm512_reduce_add_epi32(acc0) - bias;
            g_output[(i+1) * HEAD_DIM + j] = _mm512_reduce_add_epi32(acc1) - bias;
            g_output[(i+2) * HEAD_DIM + j] = _mm512_reduce_add_epi32(acc2) - bias;
            g_output[(i+3) * HEAD_DIM + j] = _mm512_reduce_add_epi32(acc3) - bias;
        }
    }

    int32_t checksum = 0;
    for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++) checksum += g_output[i];
    return checksum;
}

// C++ VNNI 4 heads
int32_t vnni_4head_noinit() {
    int32_t total = 0;
    for (int h = 0; h < 4; h++) {
        total += vnni_1head_noinit(g_Attn[h], g_V_T[h]);
    }
    return total;
}

template<typename F>
double benchmark(F func, int iterations) {
    // Warmup
    for (int i = 0; i < WARMUP; i++) func();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) func();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.so>" << std::endl;
        return 1;
    }

    // Load SimpLang kernel
    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Error loading " << argv[1] << ": " << dlerror() << std::endl;
        return 1;
    }

    auto simp_1head = (Kernel1Head)dlsym(handle, "benchmark_attn_v_noinit_1head");
    auto simp_4head = (Kernel4Head)dlsym(handle, "benchmark_attn_v_noinit_4head");

    if (!simp_1head || !simp_4head) {
        std::cerr << "Error: Could not find kernel functions" << std::endl;
        std::cerr << "  1head: " << (simp_1head ? "found" : "missing") << std::endl;
        std::cerr << "  4head: " << (simp_4head ? "found" : "missing") << std::endl;
        dlclose(handle);
        return 1;
    }

    // Initialize data ONCE (outside timing)
    std::cout << "Initializing data..." << std::endl;
    init_data();

    int64_t ops_1head = 2LL * SEQ_LEN * SEQ_LEN * HEAD_DIM;  // 134M ops
    int64_t ops_4head = 4 * ops_1head;

    std::cout << "\n";
    std::cout << "===============================================================================\n";
    std::cout << "   Attn×V MatMul Benchmark - NO INIT (pure matmul time)\n";
    std::cout << "   SEQ_LEN=" << SEQ_LEN << ", HEAD_DIM=" << HEAD_DIM << "\n";
    std::cout << "===============================================================================\n\n";

    // === Single Head ===
    std::cout << "=== Single Head (1024x1024 × 1024x64 → 1024x64) ===" << std::endl;

    // Get reference checksum
    int32_t ref_checksum = vnni_1head_noinit(g_Attn[0], g_V_T[0]);
    std::cout << "Reference checksum: " << ref_checksum << std::endl;

    // C++ VNNI
    double vnni_ms = benchmark([&]() { return vnni_1head_noinit(g_Attn[0], g_V_T[0]); }, ITERATIONS);
    double vnni_gops = (ops_1head / 1e9) / (vnni_ms / 1000.0);
    std::cout << "VNNI C++:  " << std::fixed << std::setprecision(3) << vnni_ms << " ms ("
              << std::setprecision(2) << vnni_gops << " GIOP/s)" << std::endl;

    // SimpLang - pass raw V (not transposed - SimpLang does transpose internally)
    int32_t simp_check = simp_1head(PASS_MEMREF_I8(g_Attn[0], SEQ_LEN * SEQ_LEN),
                                     PASS_MEMREF_I8(g_V[0], SEQ_LEN * HEAD_DIM));
    double simp_ms = benchmark([&]() {
        return simp_1head(PASS_MEMREF_I8(g_Attn[0], SEQ_LEN * SEQ_LEN),
                          PASS_MEMREF_I8(g_V[0], SEQ_LEN * HEAD_DIM));
    }, ITERATIONS);
    double simp_gops = (ops_1head / 1e9) / (simp_ms / 1000.0);
    double pct = (simp_gops / vnni_gops) * 100.0;
    std::cout << "SimpLang:  " << std::fixed << std::setprecision(3) << simp_ms << " ms ("
              << std::setprecision(2) << simp_gops << " GIOP/s) "
              << std::setprecision(1) << pct << "% vs VNNI" << std::endl;
    std::cout << "SimpLang checksum: " << simp_check << (simp_check == ref_checksum ? " ✓" : " ✗ MISMATCH!") << std::endl;

    // === 4 Heads ===
    std::cout << "\n=== 4 Heads ===" << std::endl;

    int32_t ref_4head = vnni_4head_noinit();
    std::cout << "Reference checksum: " << ref_4head << std::endl;

    double vnni4_ms = benchmark([&]() { return vnni_4head_noinit(); }, ITERATIONS);
    double vnni4_gops = (ops_4head / 1e9) / (vnni4_ms / 1000.0);
    std::cout << "VNNI C++:  " << std::fixed << std::setprecision(3) << vnni4_ms << " ms ("
              << std::setprecision(2) << vnni4_gops << " GIOP/s)" << std::endl;

    int32_t simp4_check = simp_4head(
        PASS_MEMREF_I8(g_Attn[0], SEQ_LEN * SEQ_LEN), PASS_MEMREF_I8(g_V[0], SEQ_LEN * HEAD_DIM),
        PASS_MEMREF_I8(g_Attn[1], SEQ_LEN * SEQ_LEN), PASS_MEMREF_I8(g_V[1], SEQ_LEN * HEAD_DIM),
        PASS_MEMREF_I8(g_Attn[2], SEQ_LEN * SEQ_LEN), PASS_MEMREF_I8(g_V[2], SEQ_LEN * HEAD_DIM),
        PASS_MEMREF_I8(g_Attn[3], SEQ_LEN * SEQ_LEN), PASS_MEMREF_I8(g_V[3], SEQ_LEN * HEAD_DIM)
    );
    double simp4_ms = benchmark([&]() {
        return simp_4head(
            PASS_MEMREF_I8(g_Attn[0], SEQ_LEN * SEQ_LEN), PASS_MEMREF_I8(g_V[0], SEQ_LEN * HEAD_DIM),
            PASS_MEMREF_I8(g_Attn[1], SEQ_LEN * SEQ_LEN), PASS_MEMREF_I8(g_V[1], SEQ_LEN * HEAD_DIM),
            PASS_MEMREF_I8(g_Attn[2], SEQ_LEN * SEQ_LEN), PASS_MEMREF_I8(g_V[2], SEQ_LEN * HEAD_DIM),
            PASS_MEMREF_I8(g_Attn[3], SEQ_LEN * SEQ_LEN), PASS_MEMREF_I8(g_V[3], SEQ_LEN * HEAD_DIM)
        );
    }, ITERATIONS);
    double simp4_gops = (ops_4head / 1e9) / (simp4_ms / 1000.0);
    double pct4 = (simp4_gops / vnni4_gops) * 100.0;
    std::cout << "SimpLang:  " << std::fixed << std::setprecision(3) << simp4_ms << " ms ("
              << std::setprecision(2) << simp4_gops << " GIOP/s) "
              << std::setprecision(1) << pct4 << "% vs VNNI" << std::endl;
    std::cout << "SimpLang checksum: " << simp4_check << (simp4_check == ref_4head ? " ✓" : " ✗ MISMATCH!") << std::endl;

    std::cout << "\n===============================================================================\n";

    dlclose(handle);
    return 0;
}
