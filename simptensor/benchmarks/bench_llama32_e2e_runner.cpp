/**
 * LLaMA 3.2-1B End-to-End Attention Benchmark
 * Simulates real transformer inference with KV cache
 *
 * Tests:
 * 1. Prefill: Process prompt tokens, fill KV cache
 * 2. Decode loop: Generate tokens one at a time with KV cache
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

constexpr int WARMUP = 2;
constexpr int ITERATIONS = 5;

// MLIR MemRef ABI
#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I32_PARAMS int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_I8(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_MEMREF_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

// Function signatures
typedef int32_t (*DecodeStepFunc)(
    MEMREF_I8_PARAMS,   // x
    MEMREF_I8_PARAMS,   // Wq_T
    MEMREF_I8_PARAMS,   // Wk_T
    MEMREF_I8_PARAMS,   // Wv_T
    MEMREF_I8_PARAMS,   // Wo_T
    MEMREF_I8_PARAMS,   // k_cache
    MEMREF_I8_PARAMS,   // v_cache
    MEMREF_I32_PARAMS,  // attn_scores
    MEMREF_I32_PARAMS,  // output
    int64_t,            // pos
    int64_t, int64_t, int64_t, int64_t, int64_t  // dim, kv_dim, n_heads, n_kv_heads, head_dim
);

typedef int32_t (*PrefillFunc)(
    MEMREF_I8_PARAMS,   // X
    MEMREF_I8_PARAMS,   // Wq_T
    MEMREF_I8_PARAMS,   // Wk_T
    MEMREF_I8_PARAMS,   // Wv_T
    MEMREF_I8_PARAMS,   // Wo_T
    MEMREF_I8_PARAMS,   // k_cache
    MEMREF_I8_PARAMS,   // v_cache
    int64_t, int64_t, int64_t  // seq_len, dim, kv_dim
);

typedef int32_t (*DecodeMatmulFunc)(
    MEMREF_I8_PARAMS,   // x
    MEMREF_I8_PARAMS,   // Wq_T
    MEMREF_I8_PARAMS,   // Wk_T
    MEMREF_I8_PARAMS,   // Wv_T
    MEMREF_I8_PARAMS,   // k_cache
    MEMREF_I8_PARAMS,   // v_cache
    int64_t,            // seq_len
    int64_t, int64_t, int64_t, int64_t, int64_t  // dim, kv_dim, n_heads, n_kv_heads, head_dim
);

// Pre-allocated buffers (all pre-transposed)
alignas(64) int8_t g_x[DIM];
alignas(64) int8_t g_X_batch[128 * DIM];
alignas(64) int8_t g_Wq_T[DIM * DIM];
alignas(64) int8_t g_Wk_T[KV_DIM * DIM];
alignas(64) int8_t g_Wv_T[KV_DIM * DIM];
alignas(64) int8_t g_Wo_T[DIM * DIM];
alignas(64) int8_t g_k_cache[MAX_SEQ_LEN * KV_DIM];
alignas(64) int8_t g_v_cache[MAX_SEQ_LEN * KV_DIM];
alignas(64) int32_t g_attn_scores[N_HEADS * MAX_SEQ_LEN];
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
    init_random_i8(g_Wq_T, DIM * DIM, 3);
    init_random_i8(g_Wk_T, KV_DIM * DIM, 4);
    init_random_i8(g_Wv_T, KV_DIM * DIM, 5);
    init_random_i8(g_Wo_T, DIM * DIM, 6);
    // Pre-fill KV cache with some data
    init_random_i8(g_k_cache, MAX_SEQ_LEN * KV_DIM, 7);
    init_random_i8(g_v_cache, MAX_SEQ_LEN * KV_DIM, 8);
    memset(g_attn_scores, 0, sizeof(g_attn_scores));
    memset(g_output, 0, sizeof(g_output));
}

// C++ VNNI reference implementation
void vnni_matmul_i8(int8_t* A, int8_t* B_T, int32_t* C, int M, int N, int K) {
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

// C++ reference: full decode step with KV cache
int32_t cpp_decode_step(int64_t pos) {
    static int32_t Q[DIM], K[KV_DIM], V[KV_DIM];

    // QKV projections
    vnni_matmul_i8(g_x, g_Wq_T, Q, 1, DIM, DIM);
    vnni_matmul_i8(g_x, g_Wk_T, K, 1, KV_DIM, DIM);
    vnni_matmul_i8(g_x, g_Wv_T, V, 1, KV_DIM, DIM);

    // Store K, V in cache
    int64_t cache_offset = pos * KV_DIM;
    for (int i = 0; i < KV_DIM; i++) {
        int32_t k_val = std::max(-128, std::min(127, K[i]));
        int32_t v_val = std::max(-128, std::min(127, V[i]));
        g_k_cache[cache_offset + i] = (int8_t)k_val;
        g_v_cache[cache_offset + i] = (int8_t)v_val;
    }

    // Multi-head attention with GQA
    int64_t seq_len = pos + 1;
    int32_t output[DIM] = {0};
    int n_rep = N_HEADS / N_KV_HEADS;

    for (int h = 0; h < N_HEADS; h++) {
        int q_offset = h * HEAD_DIM;
        int kv_head = h / n_rep;
        int kv_head_offset = kv_head * HEAD_DIM;

        // Compute attention scores
        for (int64_t t = 0; t < seq_len; t++) {
            int32_t score = 0;
            int64_t k_pos = t * KV_DIM + kv_head_offset;
            for (int d = 0; d < HEAD_DIM; d++) {
                score += Q[q_offset + d] * (int32_t)g_k_cache[k_pos + d];
            }
            score /= 8;  // scale
            g_attn_scores[h * MAX_SEQ_LEN + t] = score;
        }

        // Weighted sum of values (uniform attention for benchmark)
        for (int64_t t = 0; t < seq_len; t++) {
            int64_t v_pos = t * KV_DIM + kv_head_offset;
            for (int d = 0; d < HEAD_DIM; d++) {
                output[q_offset + d] += (int32_t)g_v_cache[v_pos + d];
            }
        }
    }

    // Checksum
    int32_t checksum = 0;
    for (int i = 0; i < DIM; i++) checksum += output[i];
    return checksum;
}

// C++ reference: prefill
int32_t cpp_prefill(int seq_len) {
    static int32_t Q[128 * DIM], K[128 * KV_DIM], V[128 * KV_DIM];

    // Batched QKV
    vnni_matmul_i8(g_X_batch, g_Wq_T, Q, seq_len, DIM, DIM);
    vnni_matmul_i8(g_X_batch, g_Wk_T, K, seq_len, KV_DIM, DIM);
    vnni_matmul_i8(g_X_batch, g_Wv_T, V, seq_len, KV_DIM, DIM);

    // Store K, V in cache
    for (int t = 0; t < seq_len; t++) {
        int cache_offset = t * KV_DIM;
        for (int i = 0; i < KV_DIM; i++) {
            int32_t k_val = std::max(-128, std::min(127, K[t * KV_DIM + i]));
            int32_t v_val = std::max(-128, std::min(127, V[t * KV_DIM + i]));
            g_k_cache[cache_offset + i] = (int8_t)k_val;
            g_v_cache[cache_offset + i] = (int8_t)v_val;
        }
    }

    return Q[0] + Q[63 * DIM + 1023] + Q[127 * DIM + 2047];
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
        std::cerr << "Error: " << dlerror() << std::endl;
        return 1;
    }

    auto simp_decode = (DecodeStepFunc)dlsym(handle, "attention_decode_step");
    auto simp_prefill = (PrefillFunc)dlsym(handle, "attention_prefill");
    auto simp_decode_matmul = (DecodeMatmulFunc)dlsym(handle, "attention_decode_matmul");

    std::cout << "Functions loaded:" << std::endl;
    std::cout << "  attention_decode_step: " << (simp_decode ? "✓" : "✗") << std::endl;
    std::cout << "  attention_prefill: " << (simp_prefill ? "✓" : "✗") << std::endl;
    std::cout << "  attention_decode_matmul: " << (simp_decode_matmul ? "✓" : "✗") << std::endl;

    init_data();

    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "   LLaMA 3.2-1B End-to-End Attention Benchmark (INT8)\n";
    std::cout << "   dim=" << DIM << ", n_heads=" << N_HEADS << ", n_kv_heads=" << N_KV_HEADS;
    std::cout << ", head_dim=" << HEAD_DIM << ", kv_dim=" << KV_DIM << "\n";
    std::cout << "================================================================================\n\n";

    // ========================================
    // Test 1: Prefill (128 tokens)
    // ========================================
    if (simp_prefill) {
        std::cout << "=== Prefill: 128 tokens ===" << std::endl;
        std::cout << "QKV: [128,2048]@[2048/512,2048] + KV cache write" << std::endl;

        // Ops: Q + K + V projections
        int64_t ops = 2LL * 128 * DIM * DIM +       // Q
                      2LL * 128 * DIM * KV_DIM * 2; // K, V
        // + cache writes (memory bound, not counted)

        double cpp_ms = benchmark([&]() { return cpp_prefill(128); }, ITERATIONS);
        double cpp_gops = (ops / 1e9) / (cpp_ms / 1000.0);

        double simp_ms = benchmark([&]() {
            return simp_prefill(
                PASS_MEMREF_I8(g_X_batch, 128 * DIM),
                PASS_MEMREF_I8(g_Wq_T, DIM * DIM),
                PASS_MEMREF_I8(g_Wk_T, KV_DIM * DIM),
                PASS_MEMREF_I8(g_Wv_T, KV_DIM * DIM),
                PASS_MEMREF_I8(g_Wo_T, DIM * DIM),
                PASS_MEMREF_I8(g_k_cache, MAX_SEQ_LEN * KV_DIM),
                PASS_MEMREF_I8(g_v_cache, MAX_SEQ_LEN * KV_DIM),
                128, DIM, KV_DIM
            );
        }, ITERATIONS);
        double simp_gops = (ops / 1e9) / (simp_ms / 1000.0);
        double pct = (simp_gops / cpp_gops) * 100.0;

        std::cout << "C++ VNNI:  " << std::fixed << std::setprecision(3) << cpp_ms << " ms ("
                  << std::setprecision(1) << cpp_gops << " GIOP/s)" << std::endl;
        std::cout << "SimpLang:  " << std::setprecision(3) << simp_ms << " ms ("
                  << std::setprecision(1) << simp_gops << " GIOP/s) "
                  << std::setprecision(1) << pct << "%" << std::endl;
        std::cout << std::endl;
    }

    // ========================================
    // Test 2: Single decode step at various positions
    // ========================================
    if (simp_decode) {
        std::cout << "=== Decode Step: Single token at various KV cache positions ===" << std::endl;

        int positions[] = {0, 63, 127, 255, 511, 1023, 1535, 2047};
        for (int pos : positions) {
            // Re-init KV cache
            init_random_i8(g_k_cache, (pos + 1) * KV_DIM, 7);
            init_random_i8(g_v_cache, (pos + 1) * KV_DIM, 8);

            // Ops: QKV + attention (Q@K^T + Attn@V per head)
            int64_t seq_len = pos + 1;
            int64_t qkv_ops = 2LL * 1 * DIM * DIM + 2LL * 1 * DIM * KV_DIM * 2;
            int64_t attn_ops = N_HEADS * (2LL * HEAD_DIM * seq_len +   // Q@K^T per head
                                          2LL * seq_len * HEAD_DIM);    // Attn@V per head
            int64_t total_ops = qkv_ops + attn_ops;

            double cpp_ms = benchmark([&]() { return cpp_decode_step(pos); }, ITERATIONS);
            double cpp_gops = (total_ops / 1e9) / (cpp_ms / 1000.0);

            double simp_ms = benchmark([&]() {
                return simp_decode(
                    PASS_MEMREF_I8(g_x, DIM),
                    PASS_MEMREF_I8(g_Wq_T, DIM * DIM),
                    PASS_MEMREF_I8(g_Wk_T, KV_DIM * DIM),
                    PASS_MEMREF_I8(g_Wv_T, KV_DIM * DIM),
                    PASS_MEMREF_I8(g_Wo_T, DIM * DIM),
                    PASS_MEMREF_I8(g_k_cache, MAX_SEQ_LEN * KV_DIM),
                    PASS_MEMREF_I8(g_v_cache, MAX_SEQ_LEN * KV_DIM),
                    PASS_MEMREF_I32(g_attn_scores, N_HEADS * MAX_SEQ_LEN),
                    PASS_MEMREF_I32(g_output, DIM),
                    pos, DIM, KV_DIM, N_HEADS, N_KV_HEADS, HEAD_DIM
                );
            }, ITERATIONS);
            double simp_gops = (total_ops / 1e9) / (simp_ms / 1000.0);
            double pct = (simp_gops / cpp_gops) * 100.0;

            std::cout << "pos=" << std::setw(4) << pos << " (seq=" << std::setw(4) << seq_len << "): "
                      << "C++=" << std::fixed << std::setprecision(3) << cpp_ms << "ms "
                      << "Simp=" << std::setprecision(3) << simp_ms << "ms "
                      << "(" << std::setprecision(1) << pct << "%)" << std::endl;
        }
        std::cout << std::endl;
    }

    // ========================================
    // Test 3: Simulated decode loop (measure tokens/sec)
    // ========================================
    if (simp_decode) {
        std::cout << "=== Decode Loop Simulation ===" << std::endl;

        // Test different starting positions (simulating different prompt lengths)
        int test_configs[][2] = {
            {128, 128},    // Short prompt, generate 128 tokens
            {512, 256},    // Medium prompt, generate 256 tokens
            {1024, 512},   // Long prompt, generate 512 tokens
            {1536, 512},   // Very long prompt, generate to max
        };

        for (auto& config : test_configs) {
            int start_pos = config[0];
            int num_tokens = config[1];

            // Clamp to max seq len
            if (start_pos + num_tokens > MAX_SEQ_LEN) {
                num_tokens = MAX_SEQ_LEN - start_pos;
            }
            if (num_tokens <= 0) continue;

            // Re-init cache for prefill
            init_random_i8(g_k_cache, start_pos * KV_DIM, 7);
            init_random_i8(g_v_cache, start_pos * KV_DIM, 8);

            // C++ decode loop
            auto cpp_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_tokens; i++) {
                cpp_decode_step(start_pos + i);
            }
            auto cpp_end = std::chrono::high_resolution_clock::now();
            double cpp_total_ms = std::chrono::duration<double, std::milli>(cpp_end - cpp_start).count();
            double cpp_tokens_per_sec = num_tokens / (cpp_total_ms / 1000.0);

            // Re-init cache
            init_random_i8(g_k_cache, start_pos * KV_DIM, 7);
            init_random_i8(g_v_cache, start_pos * KV_DIM, 8);

            // SimpLang decode loop
            auto simp_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_tokens; i++) {
                simp_decode(
                    PASS_MEMREF_I8(g_x, DIM),
                    PASS_MEMREF_I8(g_Wq_T, DIM * DIM),
                    PASS_MEMREF_I8(g_Wk_T, KV_DIM * DIM),
                    PASS_MEMREF_I8(g_Wv_T, KV_DIM * DIM),
                    PASS_MEMREF_I8(g_Wo_T, DIM * DIM),
                    PASS_MEMREF_I8(g_k_cache, MAX_SEQ_LEN * KV_DIM),
                    PASS_MEMREF_I8(g_v_cache, MAX_SEQ_LEN * KV_DIM),
                    PASS_MEMREF_I32(g_attn_scores, N_HEADS * MAX_SEQ_LEN),
                    PASS_MEMREF_I32(g_output, DIM),
                    start_pos + i, DIM, KV_DIM, N_HEADS, N_KV_HEADS, HEAD_DIM
                );
            }
            auto simp_end = std::chrono::high_resolution_clock::now();
            double simp_total_ms = std::chrono::duration<double, std::milli>(simp_end - simp_start).count();
            double simp_tokens_per_sec = num_tokens / (simp_total_ms / 1000.0);

            std::cout << "pos=" << std::setw(4) << start_pos << " -> " << std::setw(4) << (start_pos + num_tokens)
                      << " (" << std::setw(3) << num_tokens << " tokens): "
                      << "C++=" << std::fixed << std::setprecision(1) << cpp_tokens_per_sec << " tok/s  "
                      << "Simp=" << std::setprecision(1) << simp_tokens_per_sec << " tok/s  "
                      << "(" << std::setprecision(1) << (simp_tokens_per_sec / cpp_tokens_per_sec * 100) << "%)"
                      << std::endl;
        }
        std::cout << std::endl;
    }

    // ========================================
    // Test 4: QKV + cache write only (decode_matmul)
    // ========================================
    if (simp_decode_matmul) {
        std::cout << "=== QKV + KV Cache Write (decode step, no attention) ===" << std::endl;

        int64_t ops = 2LL * 1 * DIM * DIM + 2LL * 1 * DIM * KV_DIM * 2;

        double simp_ms = benchmark([&]() {
            return simp_decode_matmul(
                PASS_MEMREF_I8(g_x, DIM),
                PASS_MEMREF_I8(g_Wq_T, DIM * DIM),
                PASS_MEMREF_I8(g_Wk_T, KV_DIM * DIM),
                PASS_MEMREF_I8(g_Wv_T, KV_DIM * DIM),
                PASS_MEMREF_I8(g_k_cache, MAX_SEQ_LEN * KV_DIM),
                PASS_MEMREF_I8(g_v_cache, MAX_SEQ_LEN * KV_DIM),
                512,  // seq_len
                DIM, KV_DIM, N_HEADS, N_KV_HEADS, HEAD_DIM
            );
        }, ITERATIONS);
        double simp_gops = (ops / 1e9) / (simp_ms / 1000.0);

        std::cout << "SimpLang:  " << std::fixed << std::setprecision(3) << simp_ms << " ms ("
                  << std::setprecision(1) << simp_gops << " GIOP/s)" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "================================================================================\n";

    dlclose(handle);
    return 0;
}
