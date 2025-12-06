/**
 * LLaMA 3.2-1B Attention Block Benchmark Runner
 * INT8 weights, INT32 accumulation
 *
 * Config:
 *   dim = 2048
 *   n_heads = 32 (query heads)
 *   n_kv_heads = 8 (GQA)
 *   head_dim = 64
 *   kv_dim = 512
 */

#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cstring>
#include <immintrin.h>
#include <random>

// LLaMA 3.2-1B config
constexpr int DIM = 2048;
constexpr int N_HEADS = 32;
constexpr int N_KV_HEADS = 8;
constexpr int HEAD_DIM = 64;
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;  // 512
constexpr int MAX_SEQ_LEN = 2048;

constexpr int WARMUP = 3;
constexpr int ITERATIONS = 10;

// MLIR MemRef ABI
#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I32_PARAMS int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_I8(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_MEMREF_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

// Function signatures
typedef int32_t (*QKVProjectionFunc)(
    MEMREF_I8_PARAMS,  // x
    MEMREF_I8_PARAMS,  // Wq
    MEMREF_I8_PARAMS,  // Wk
    MEMREF_I8_PARAMS,  // Wv
    int64_t, int64_t   // dim, kv_dim
);

typedef int32_t (*QKVBatchedFunc)(
    MEMREF_I8_PARAMS,  // X
    MEMREF_I8_PARAMS,  // Wq
    MEMREF_I8_PARAMS,  // Wk
    MEMREF_I8_PARAMS,  // Wv
    int64_t, int64_t, int64_t  // batch, dim, kv_dim
);

typedef int32_t (*AttnScoresFunc)(
    MEMREF_I8_PARAMS,  // Q
    MEMREF_I8_PARAMS,  // K_cache
    int64_t, int64_t   // seq_len, head_dim
);

typedef int32_t (*AttnOutputFunc)(
    MEMREF_I8_PARAMS,  // attn_weights
    MEMREF_I8_PARAMS,  // V_cache
    int64_t, int64_t   // seq_len, head_dim
);

// Pre-allocated buffers - ALL WEIGHTS PRE-TRANSPOSED
alignas(64) int8_t g_x[DIM];
alignas(64) int8_t g_X_batch[128 * DIM];  // For batched prefill
alignas(64) int8_t g_Wq_T[DIM * DIM];     // [N=2048, K=2048] pre-transposed
alignas(64) int8_t g_Wk_T[KV_DIM * DIM];  // [N=512, K=2048] pre-transposed
alignas(64) int8_t g_Wv_T[KV_DIM * DIM];  // [N=512, K=2048] pre-transposed
alignas(64) int8_t g_Wo_T[DIM * DIM];     // [N=2048, K=2048] pre-transposed
alignas(64) int8_t g_Q[1024 * HEAD_DIM];
alignas(64) int8_t g_K_cache[1024 * HEAD_DIM];  // [N=1024, K=64] for matmul_nt
alignas(64) int8_t g_attn[1024 * 1024];
alignas(64) int8_t g_V_T[HEAD_DIM * 1024];      // [N=64, K=1024] pre-transposed
alignas(64) int32_t g_output[DIM];

void init_random_i8(int8_t* data, size_t size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-64, 63);
    for (size_t i = 0; i < size; i++) {
        data[i] = (int8_t)dist(rng);
    }
}

void init_data() {
    init_random_i8(g_x, DIM, 1);
    init_random_i8(g_X_batch, 128 * DIM, 2);
    // All weights are directly initialized as pre-transposed [N, K]
    init_random_i8(g_Wq_T, DIM * DIM, 3);       // [N=2048, K=2048]
    init_random_i8(g_Wk_T, KV_DIM * DIM, 4);    // [N=512, K=2048]
    init_random_i8(g_Wv_T, KV_DIM * DIM, 5);    // [N=512, K=2048]
    init_random_i8(g_Wo_T, DIM * DIM, 6);       // [N=2048, K=2048]
    init_random_i8(g_Q, 1024 * HEAD_DIM, 7);
    init_random_i8(g_K_cache, 1024 * HEAD_DIM, 8);  // [N=1024, K=64] already in correct layout
    init_random_i8(g_attn, 1024 * 1024, 9);
    init_random_i8(g_V_T, HEAD_DIM * 1024, 10);     // [N=64, K=1024]
}

// C++ VNNI reference: matmul with I=4 tiling
void vnni_matmul_i8(int8_t* A, int8_t* B_T, int32_t* C, int M, int N, int K) {
    // A: [M, K], B_T: [N, K] (transposed), C: [M, N]
    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);

    memset(C, 0, M * N * sizeof(int32_t));

    for (int i = 0; i < M; i += 4) {
        int i_end = std::min(i + 4, M);
        for (int j = 0; j < N; j++) {
            __m512i acc[4] = {_mm512_setzero_si512()};
            __m512i bias_acc = _mm512_setzero_si512();

            for (int k = 0; k < K; k += 64) {
                __m512i b = _mm512_loadu_si512(&B_T[j * K + k]);

                for (int ii = 0; ii < i_end - i; ii++) {
                    __m512i a = _mm512_xor_si512(
                        _mm512_loadu_si512(&A[(i + ii) * K + k]), sign_flip);
                    acc[ii] = _mm512_dpbusd_epi32(acc[ii], a, b);
                }
                bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, b);
            }

            int32_t bias = _mm512_reduce_add_epi32(bias_acc) * 128;
            for (int ii = 0; ii < i_end - i; ii++) {
                C[(i + ii) * N + j] = _mm512_reduce_add_epi32(acc[ii]) - bias;
            }
        }
    }
}

// C++ VNNI QKV projection - using pre-transposed weights
int32_t vnni_qkv_projection() {
    static int32_t Q[DIM], K[KV_DIM], V[KV_DIM];

    // Q = x @ Wq_T: [1, 2048] @ [2048, 2048]_T -> [1, 2048]
    vnni_matmul_i8(g_x, g_Wq_T, Q, 1, DIM, DIM);

    // K = x @ Wk_T: [1, 2048] @ [512, 2048]_T -> [1, 512]
    vnni_matmul_i8(g_x, g_Wk_T, K, 1, KV_DIM, DIM);

    // V = x @ Wv_T: [1, 2048] @ [512, 2048]_T -> [1, 512]
    vnni_matmul_i8(g_x, g_Wv_T, V, 1, KV_DIM, DIM);

    int32_t checksum = 0;
    for (int i = 0; i < DIM; i++) checksum += Q[i];
    for (int i = 0; i < KV_DIM; i++) checksum += K[i] + V[i];
    return checksum;
}

// C++ VNNI attention scores: Q @ K^T - using pre-transposed K_cache
int32_t vnni_attention_scores() {
    static int32_t scores[1024 * 1024];
    // g_K_cache is [N=1024, K=64] - already in the right layout for matmul_nt
    vnni_matmul_i8(g_Q, g_K_cache, scores, 1024, 1024, HEAD_DIM);
    return scores[0] + scores[511 * 1024 + 511] + scores[1023 * 1024 + 1023];
}

// C++ VNNI attention output: Attn @ V - using pre-transposed V
int32_t vnni_attention_output() {
    static int32_t output[1024 * HEAD_DIM];
    // g_V_T is [N=64, K=1024] - already in the right layout for matmul_nt
    vnni_matmul_i8(g_attn, g_V_T, output, 1024, HEAD_DIM, 1024);
    return output[0] + output[511 * HEAD_DIM + 31] + output[1023 * HEAD_DIM + 63];
}

template<typename F>
double benchmark(F func, int iterations) {
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

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Error loading " << argv[1] << ": " << dlerror() << std::endl;
        return 1;
    }

    auto simp_qkv = (QKVProjectionFunc)dlsym(handle, "bench_qkv_projection");
    auto simp_qkv_batched = (QKVBatchedFunc)dlsym(handle, "bench_qkv_batched");
    auto simp_attn_scores = (AttnScoresFunc)dlsym(handle, "bench_attention_scores");
    auto simp_attn_output = (AttnOutputFunc)dlsym(handle, "bench_attention_output");

    std::cout << "Functions loaded:" << std::endl;
    std::cout << "  bench_qkv_projection: " << (simp_qkv ? "✓" : "✗") << std::endl;
    std::cout << "  bench_qkv_batched: " << (simp_qkv_batched ? "✓" : "✗") << std::endl;
    std::cout << "  bench_attention_scores: " << (simp_attn_scores ? "✓" : "✗") << std::endl;
    std::cout << "  bench_attention_output: " << (simp_attn_output ? "✓" : "✗") << std::endl;

    std::cout << "\nInitializing data..." << std::endl;
    init_data();

    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "   LLaMA 3.2-1B Attention Block Benchmark (INT8)\n";
    std::cout << "   dim=" << DIM << ", n_heads=" << N_HEADS << ", n_kv_heads=" << N_KV_HEADS;
    std::cout << ", head_dim=" << HEAD_DIM << "\n";
    std::cout << "================================================================================\n\n";

    // ========================================
    // Benchmark 1: QKV Projection (single token)
    // ========================================
    if (simp_qkv) {
        std::cout << "=== QKV Projection (single token decode) ===" << std::endl;
        std::cout << "Q: [1,2048]@[2048,2048], K/V: [1,2048]@[2048,512]" << std::endl;

        int64_t ops = 2LL * 1 * DIM * DIM +       // Q projection
                      2LL * 1 * DIM * KV_DIM * 2; // K, V projections
        // ops = 2*2048*2048 + 2*2*2048*512 = 8.4M + 4.2M = 12.6M ops

        double vnni_ms = benchmark(vnni_qkv_projection, ITERATIONS);
        double vnni_gops = (ops / 1e9) / (vnni_ms / 1000.0);
        std::cout << "VNNI C++:  " << std::fixed << std::setprecision(3) << vnni_ms << " ms ("
                  << std::setprecision(2) << vnni_gops << " GIOP/s)" << std::endl;

        int32_t simp_check = simp_qkv(
            PASS_MEMREF_I8(g_x, DIM),
            PASS_MEMREF_I8(g_Wq_T, DIM * DIM),
            PASS_MEMREF_I8(g_Wk_T, KV_DIM * DIM),
            PASS_MEMREF_I8(g_Wv_T, KV_DIM * DIM),
            DIM, KV_DIM
        );
        double simp_ms = benchmark([&]() {
            return simp_qkv(
                PASS_MEMREF_I8(g_x, DIM),
                PASS_MEMREF_I8(g_Wq_T, DIM * DIM),
                PASS_MEMREF_I8(g_Wk_T, KV_DIM * DIM),
                PASS_MEMREF_I8(g_Wv_T, KV_DIM * DIM),
                DIM, KV_DIM
            );
        }, ITERATIONS);
        double simp_gops = (ops / 1e9) / (simp_ms / 1000.0);
        double pct = (simp_gops / vnni_gops) * 100.0;
        std::cout << "SimpLang:  " << std::fixed << std::setprecision(3) << simp_ms << " ms ("
                  << std::setprecision(2) << simp_gops << " GIOP/s) "
                  << std::setprecision(1) << pct << "% vs VNNI" << std::endl;
        std::cout << std::endl;
    }

    // ========================================
    // Benchmark 2: Attention Scores Q@K^T
    // ========================================
    if (simp_attn_scores) {
        std::cout << "=== Attention Scores Q@K^T (per head, seq_len=1024) ===" << std::endl;
        std::cout << "[1024,64] @ [64,1024] -> [1024,1024]" << std::endl;

        int64_t ops = 2LL * 1024 * 1024 * HEAD_DIM;  // 134M ops

        double vnni_ms = benchmark(vnni_attention_scores, ITERATIONS);
        double vnni_gops = (ops / 1e9) / (vnni_ms / 1000.0);
        std::cout << "VNNI C++:  " << std::fixed << std::setprecision(3) << vnni_ms << " ms ("
                  << std::setprecision(2) << vnni_gops << " GIOP/s)" << std::endl;

        int32_t simp_check = simp_attn_scores(
            PASS_MEMREF_I8(g_Q, 1024 * HEAD_DIM),
            PASS_MEMREF_I8(g_K_cache, 1024 * HEAD_DIM),
            1024, HEAD_DIM
        );
        double simp_ms = benchmark([&]() {
            return simp_attn_scores(
                PASS_MEMREF_I8(g_Q, 1024 * HEAD_DIM),
                PASS_MEMREF_I8(g_K_cache, 1024 * HEAD_DIM),
                1024, HEAD_DIM
            );
        }, ITERATIONS);
        double simp_gops = (ops / 1e9) / (simp_ms / 1000.0);
        double pct = (simp_gops / vnni_gops) * 100.0;
        std::cout << "SimpLang:  " << std::fixed << std::setprecision(3) << simp_ms << " ms ("
                  << std::setprecision(2) << simp_gops << " GIOP/s) "
                  << std::setprecision(1) << pct << "% vs VNNI" << std::endl;
        std::cout << std::endl;
    }

    // ========================================
    // Benchmark 3: Attention Output Attn@V
    // ========================================
    if (simp_attn_output) {
        std::cout << "=== Attention Output Attn@V (per head, seq_len=1024) ===" << std::endl;
        std::cout << "[1024,1024] @ [1024,64] -> [1024,64]" << std::endl;

        int64_t ops = 2LL * 1024 * 1024 * HEAD_DIM;  // 134M ops

        double vnni_ms = benchmark(vnni_attention_output, ITERATIONS);
        double vnni_gops = (ops / 1e9) / (vnni_ms / 1000.0);
        std::cout << "VNNI C++:  " << std::fixed << std::setprecision(3) << vnni_ms << " ms ("
                  << std::setprecision(2) << vnni_gops << " GIOP/s)" << std::endl;

        int32_t simp_check = simp_attn_output(
            PASS_MEMREF_I8(g_attn, 1024 * 1024),
            PASS_MEMREF_I8(g_V_T, HEAD_DIM * 1024),
            1024, HEAD_DIM
        );
        double simp_ms = benchmark([&]() {
            return simp_attn_output(
                PASS_MEMREF_I8(g_attn, 1024 * 1024),
                PASS_MEMREF_I8(g_V_T, HEAD_DIM * 1024),
                1024, HEAD_DIM
            );
        }, ITERATIONS);
        double simp_gops = (ops / 1e9) / (simp_ms / 1000.0);
        double pct = (simp_gops / vnni_gops) * 100.0;
        std::cout << "SimpLang:  " << std::fixed << std::setprecision(3) << simp_ms << " ms ("
                  << std::setprecision(2) << simp_gops << " GIOP/s) "
                  << std::setprecision(1) << pct << "% vs VNNI" << std::endl;
        std::cout << std::endl;
    }

    // ========================================
    // Benchmark 4: Batched QKV (prefill)
    // ========================================
    if (simp_qkv_batched) {
        std::cout << "=== Batched QKV Projection (prefill, batch=128) ===" << std::endl;
        std::cout << "Q: [128,2048]@[2048,2048], K/V: [128,2048]@[2048,512]" << std::endl;

        int64_t ops = 2LL * 128 * DIM * DIM +       // Q
                      2LL * 128 * DIM * KV_DIM * 2; // K, V
        // = 128 * 12.6M = 1.6B ops

        int32_t simp_check = simp_qkv_batched(
            PASS_MEMREF_I8(g_X_batch, 128 * DIM),
            PASS_MEMREF_I8(g_Wq_T, DIM * DIM),
            PASS_MEMREF_I8(g_Wk_T, KV_DIM * DIM),
            PASS_MEMREF_I8(g_Wv_T, KV_DIM * DIM),
            128, DIM, KV_DIM
        );
        double simp_ms = benchmark([&]() {
            return simp_qkv_batched(
                PASS_MEMREF_I8(g_X_batch, 128 * DIM),
                PASS_MEMREF_I8(g_Wq_T, DIM * DIM),
                PASS_MEMREF_I8(g_Wk_T, KV_DIM * DIM),
                PASS_MEMREF_I8(g_Wv_T, KV_DIM * DIM),
                128, DIM, KV_DIM
            );
        }, ITERATIONS);
        double simp_gops = (ops / 1e9) / (simp_ms / 1000.0);
        std::cout << "SimpLang:  " << std::fixed << std::setprecision(3) << simp_ms << " ms ("
                  << std::setprecision(2) << simp_gops << " GIOP/s)" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "================================================================================\n";

    dlclose(handle);
    return 0;
}
