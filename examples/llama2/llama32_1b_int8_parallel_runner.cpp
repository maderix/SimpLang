// LLaMA 3.2-1B Parallel Prefill Runner
// Uses OpenMP to parallelize matmul operations
// Compile: g++ -O3 -march=native -fopenmp -mavx512vnni -o /tmp/bench_1b_parallel llama32_1b_int8_parallel_runner.cpp -ldl

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <dlfcn.h>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

// Model config
constexpr int64_t DIM = 2048;
constexpr int64_t HIDDEN_DIM = 8192;
constexpr int64_t N_LAYERS = 16;
constexpr int64_t N_HEADS = 32;
constexpr int64_t N_KV_HEADS = 8;
constexpr int64_t HEAD_DIM = 64;
constexpr int64_t KV_DIM = N_KV_HEADS * HEAD_DIM;
constexpr int64_t MAX_SEQ_LEN = 4096;

// Parallel VNNI INT8 MatMul: C[M,N] = A[M,K] @ B_T[N,K]
// Parallelizes the M (row) loop
void matmul_int8_parallel(
    const int8_t* A,      // [M, K]
    const int8_t* B_T,    // [N, K] (transposed)
    int32_t* C,           // [M, N]
    int M, int N, int K
) {
    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i += 4) {
        int rows = std::min(4, M - i);
        for (int j = 0; j < N; j++) {
            __m512i acc[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(),
                             _mm512_setzero_si512(), _mm512_setzero_si512()};
            __m512i bias_acc = _mm512_setzero_si512();

            for (int k = 0; k + 64 <= K; k += 64) {
                __m512i vb = _mm512_loadu_si512(&B_T[j * K + k]);

                for (int r = 0; r < rows; r++) {
                    __m512i va = _mm512_xor_si512(
                        _mm512_loadu_si512(&A[(i + r) * K + k]), sign_flip);
                    acc[r] = _mm512_dpbusd_epi32(acc[r], va, vb);
                }
                bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, vb);
            }

            // Handle remainder
            int rem = K % 64;
            if (rem > 0) {
                int k = K - rem;
                __mmask64 mask = (1ULL << rem) - 1;
                __m512i vb = _mm512_maskz_loadu_epi8(mask, &B_T[j * K + k]);
                for (int r = 0; r < rows; r++) {
                    __m512i va = _mm512_xor_si512(
                        _mm512_maskz_loadu_epi8(mask, &A[(i + r) * K + k]), sign_flip);
                    acc[r] = _mm512_dpbusd_epi32(acc[r], va, vb);
                }
                bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, vb);
            }

            int32_t bias = _mm512_reduce_add_epi32(bias_acc);
            int32_t correction = bias * 128;

            for (int r = 0; r < rows; r++) {
                C[(i + r) * N + j] = _mm512_reduce_add_epi32(acc[r]) - correction;
            }
        }
    }
}

// Single-threaded version for comparison
void matmul_int8_single(
    const int8_t* A, const int8_t* B_T, int32_t* C,
    int M, int N, int K
) {
    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    const __m512i ones = _mm512_set1_epi8(1);

    for (int i = 0; i < M; i += 4) {
        int rows = std::min(4, M - i);
        for (int j = 0; j < N; j++) {
            __m512i acc[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(),
                             _mm512_setzero_si512(), _mm512_setzero_si512()};
            __m512i bias_acc = _mm512_setzero_si512();

            for (int k = 0; k + 64 <= K; k += 64) {
                __m512i vb = _mm512_loadu_si512(&B_T[j * K + k]);
                for (int r = 0; r < rows; r++) {
                    __m512i va = _mm512_xor_si512(
                        _mm512_loadu_si512(&A[(i + r) * K + k]), sign_flip);
                    acc[r] = _mm512_dpbusd_epi32(acc[r], va, vb);
                }
                bias_acc = _mm512_dpbusd_epi32(bias_acc, ones, vb);
            }

            int32_t bias = _mm512_reduce_add_epi32(bias_acc);
            int32_t correction = bias * 128;
            for (int r = 0; r < rows; r++) {
                C[(i + r) * N + j] = _mm512_reduce_add_epi32(acc[r]) - correction;
            }
        }
    }
}

// RMSNorm (simple, not parallelized - small op)
void rmsnorm_i16(const int16_t* x, const int16_t* w, int16_t* out, int dim) {
    int64_t ss = 0;
    for (int i = 0; i < dim; i++) {
        int32_t v = x[i];
        ss += v * v;
    }
    ss = ss / dim + 1;

    // Newton-Raphson rsqrt
    int64_t y = 256;
    for (int j = 0; j < 4; j++) {
        int64_t y2 = y * y;
        int64_t xy2 = (ss * y2) / 65536;
        int64_t term = 196608 - xy2;
        y = (y * term) / 131072;
    }

    for (int i = 0; i < dim; i++) {
        int32_t normed = (x[i] * y) / 256;
        int32_t result = (normed * w[i]) / 32768;
        out[i] = (int16_t)std::max(-32768, std::min(32767, result));
    }
}

// Quantize i32 -> i8
void quantize_i32_to_i8(const int32_t* in, int8_t* out, int n, int scale) {
    for (int i = 0; i < n; i++) {
        int32_t v = in[i] / scale;
        out[i] = (int8_t)std::max(-128, std::min(127, v));
    }
}

// Quantize i16 -> i8
void quantize_i16_to_i8(const int16_t* in, int8_t* out, int n, int shift) {
    for (int i = 0; i < n; i++) {
        int32_t v = in[i] >> shift;
        out[i] = (int8_t)std::max(-128, std::min(127, v));
    }
}

// SiLU element-wise: out = silu(gate) * up
void silu_mul(const int32_t* gate, const int32_t* up, int8_t* out, int n, int scale) {
    for (int i = 0; i < n; i++) {
        int32_t g = gate[i] / 512;
        int32_t u = up[i] / 512;

        // Sigmoid approximation
        int32_t linear = g / 4;
        int32_t correction = (g * g) / 8192;
        int32_t sig = 128 + linear;
        if (g > 0) sig -= correction;
        if (g < 0) sig += correction;
        sig = std::max(0, std::min(256, sig));

        int32_t silu = (g * sig) / 256;
        int32_t result = (silu * u) / scale;
        out[i] = (int8_t)std::max(-128, std::min(127, result));
    }
}

// Softmax i32 -> i8
void softmax(const int32_t* in, int8_t* out, int n) {
    int32_t max_val = in[0];
    for (int i = 1; i < n; i++) {
        if (in[i] > max_val) max_val = in[i];
    }

    int32_t sum = 0;
    for (int i = 0; i < n; i++) {
        int32_t diff = in[i] - max_val;
        if (diff < -127) diff = -127;
        int32_t e = 128 + diff * 8 + (diff * diff) / 4;
        e = std::max(1, std::min(255, e));
        out[i] = (int8_t)e;
        sum += e;
    }

    if (sum > 0) {
        for (int i = 0; i < n; i++) {
            int32_t p = ((int32_t)(uint8_t)out[i]) * 127 / sum;
            out[i] = (int8_t)p;
        }
    }
}

// RoPE
void rope(int32_t* q, int32_t* k, const int16_t* cos_tab, const int16_t* sin_tab,
          int dim, int kv_dim, int pos) {
    int half_head = HEAD_DIM / 2;
    int pos_off = pos * half_head;

    // Q
    for (int h = 0; h < N_HEADS; h++) {
        for (int i = 0; i < half_head; i++) {
            int32_t q0 = q[h * HEAD_DIM + i * 2];
            int32_t q1 = q[h * HEAD_DIM + i * 2 + 1];
            int32_t c = cos_tab[pos_off + i];
            int32_t s = sin_tab[pos_off + i];
            q[h * HEAD_DIM + i * 2] = (q0 * c - q1 * s) / 32768;
            q[h * HEAD_DIM + i * 2 + 1] = (q0 * s + q1 * c) / 32768;
        }
    }

    // K
    for (int h = 0; h < N_KV_HEADS; h++) {
        for (int i = 0; i < half_head; i++) {
            int32_t k0 = k[h * HEAD_DIM + i * 2];
            int32_t k1 = k[h * HEAD_DIM + i * 2 + 1];
            int32_t c = cos_tab[pos_off + i];
            int32_t s = sin_tab[pos_off + i];
            k[h * HEAD_DIM + i * 2] = (k0 * c - k1 * s) / 32768;
            k[h * HEAD_DIM + i * 2 + 1] = (k0 * s + k1 * c) / 32768;
        }
    }
}

// Aligned allocation
template<typename T>
T* alloc_aligned(size_t count) {
    return (T*)aligned_alloc(64, count * sizeof(T));
}

void init_random_i8(int8_t* ptr, size_t n) {
    for (size_t i = 0; i < n; i++) ptr[i] = (int8_t)((rand() % 256) - 128);
}

void init_random_i16(int16_t* ptr, size_t n) {
    for (size_t i = 0; i < n; i++) ptr[i] = (int16_t)((rand() % 65536) - 32768);
}

void init_rope_tables(int16_t* cos_tab, int16_t* sin_tab, int max_seq, int half_head) {
    float theta = 500000.0f;
    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < half_head; i++) {
            float freq = 1.0f / powf(theta, (float)(i * 2) / (float)HEAD_DIM);
            float angle = (float)pos * freq;
            cos_tab[pos * half_head + i] = (int16_t)(cosf(angle) * 32767.0f);
            sin_tab[pos * half_head + i] = (int16_t)(sinf(angle) * 32767.0f);
        }
    }
}

// Forward one token through all layers (parallel matmuls)
void forward_token_parallel(
    int8_t* x,                    // [DIM]
    int16_t* rms_att_w,           // [N_LAYERS, DIM]
    int16_t* rms_ffn_w,           // [N_LAYERS, DIM]
    int8_t* wq_t, int8_t* wk_t, int8_t* wv_t, int8_t* wo_t,
    int8_t* w1_t, int8_t* w2_t, int8_t* w3_t,
    int8_t* k_cache, int8_t* v_cache,
    int16_t* cos_tab, int16_t* sin_tab,
    // Temp buffers
    int16_t* xb_i16, int8_t* xb_i8,
    int32_t* q_i32, int32_t* k_i32, int32_t* v_i32,
    int32_t* attn_out, int32_t* att_scores, int8_t* att_probs,
    int32_t* ffn_gate, int32_t* ffn_up, int8_t* ffn_hb,
    int pos
) {
    // Convert x to i16 for RMSNorm
    for (int i = 0; i < DIM; i++) xb_i16[i] = x[i] * 128;

    for (int layer = 0; layer < N_LAYERS; layer++) {
        int64_t l_off = layer * DIM;

        // RMSNorm for attention
        int16_t* norm_out = xb_i16 + DIM;  // Use second half as temp
        rmsnorm_i16(xb_i16, rms_att_w + l_off, norm_out, DIM);
        quantize_i16_to_i8(norm_out, xb_i8, DIM, 8);

        // Q, K, V projections (parallel matmuls)
        matmul_int8_parallel(xb_i8, wq_t + layer * DIM * DIM, q_i32, 1, DIM, DIM);
        matmul_int8_parallel(xb_i8, wk_t + layer * KV_DIM * DIM, k_i32, 1, KV_DIM, DIM);
        matmul_int8_parallel(xb_i8, wv_t + layer * KV_DIM * DIM, v_i32, 1, KV_DIM, DIM);

        // RoPE
        rope(q_i32, k_i32, cos_tab, sin_tab, DIM, KV_DIM, pos);

        // Store K, V in cache
        int64_t cache_off = layer * MAX_SEQ_LEN * KV_DIM + pos * KV_DIM;
        quantize_i32_to_i8(k_i32, k_cache + cache_off, KV_DIM, 512);
        quantize_i32_to_i8(v_i32, v_cache + cache_off, KV_DIM, 512);

        // Attention (simplified for decode - uses full cache up to pos)
        int seq_len = pos + 1;
        memset(attn_out, 0, DIM * sizeof(int32_t));

        for (int h = 0; h < N_HEADS; h++) {
            int kv_h = h / (N_HEADS / N_KV_HEADS);

            // Q @ K^T
            for (int t = 0; t < seq_len; t++) {
                int32_t score = 0;
                int64_t k_off = layer * MAX_SEQ_LEN * KV_DIM + t * KV_DIM + kv_h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) {
                    score += q_i32[h * HEAD_DIM + d] * (int32_t)k_cache[k_off + d];
                }
                att_scores[h * seq_len + t] = score / 8;  // Scale
            }

            // Softmax
            softmax(att_scores + h * seq_len, att_probs + h * seq_len, seq_len);

            // Attention @ V
            for (int d = 0; d < HEAD_DIM; d++) {
                int32_t sum = 0;
                for (int t = 0; t < seq_len; t++) {
                    int64_t v_off = layer * MAX_SEQ_LEN * KV_DIM + t * KV_DIM + kv_h * HEAD_DIM;
                    sum += (int32_t)(uint8_t)att_probs[h * seq_len + t] * (int32_t)v_cache[v_off + d];
                }
                attn_out[h * HEAD_DIM + d] = sum;
            }
        }

        // Output projection (parallel)
        quantize_i32_to_i8(attn_out, xb_i8, DIM, 256);
        int32_t* proj_out = ffn_gate;  // Reuse buffer
        matmul_int8_parallel(xb_i8, wo_t + layer * DIM * DIM, proj_out, 1, DIM, DIM);

        // Residual add
        for (int i = 0; i < DIM; i++) {
            xb_i16[i] = xb_i16[i] + proj_out[i] / 32;
        }

        // FFN
        rmsnorm_i16(xb_i16, rms_ffn_w + l_off, norm_out, DIM);
        quantize_i16_to_i8(norm_out, xb_i8, DIM, 8);

        // Gate, Up projections (parallel)
        matmul_int8_parallel(xb_i8, w1_t + layer * HIDDEN_DIM * DIM, ffn_gate, 1, HIDDEN_DIM, DIM);
        matmul_int8_parallel(xb_i8, w3_t + layer * HIDDEN_DIM * DIM, ffn_up, 1, HIDDEN_DIM, DIM);

        // SiLU * up
        silu_mul(ffn_gate, ffn_up, ffn_hb, HIDDEN_DIM, 1024);

        // Down projection (parallel)
        matmul_int8_parallel(ffn_hb, w2_t + layer * DIM * HIDDEN_DIM, proj_out, 1, DIM, HIDDEN_DIM);

        // Residual add
        for (int i = 0; i < DIM; i++) {
            xb_i16[i] = xb_i16[i] + proj_out[i] / 32;
        }
    }

    // Output to x
    quantize_i16_to_i8(xb_i16, x, DIM, 8);
}

int main(int argc, char** argv) {
    int num_threads = omp_get_max_threads();

    std::cout << "=== LLaMA 3.2-1B Parallel INT8 Benchmark ===" << std::endl;
    std::cout << "OpenMP threads: " << num_threads << std::endl << std::endl;

    srand(42);

    // Allocate
    std::cout << "Allocating memory..." << std::endl;
    auto rms_att_w = alloc_aligned<int16_t>(N_LAYERS * DIM);
    auto rms_ffn_w = alloc_aligned<int16_t>(N_LAYERS * DIM);
    auto wq_t = alloc_aligned<int8_t>(N_LAYERS * DIM * DIM);
    auto wk_t = alloc_aligned<int8_t>(N_LAYERS * KV_DIM * DIM);
    auto wv_t = alloc_aligned<int8_t>(N_LAYERS * KV_DIM * DIM);
    auto wo_t = alloc_aligned<int8_t>(N_LAYERS * DIM * DIM);
    auto w1_t = alloc_aligned<int8_t>(N_LAYERS * HIDDEN_DIM * DIM);
    auto w2_t = alloc_aligned<int8_t>(N_LAYERS * DIM * HIDDEN_DIM);
    auto w3_t = alloc_aligned<int8_t>(N_LAYERS * HIDDEN_DIM * DIM);

    auto x = alloc_aligned<int8_t>(DIM);
    auto xb_i16 = alloc_aligned<int16_t>(DIM * 2);
    auto xb_i8 = alloc_aligned<int8_t>(DIM);
    auto q_i32 = alloc_aligned<int32_t>(DIM);
    auto k_i32 = alloc_aligned<int32_t>(KV_DIM);
    auto v_i32 = alloc_aligned<int32_t>(KV_DIM);
    auto attn_out = alloc_aligned<int32_t>(DIM);
    auto att_scores = alloc_aligned<int32_t>(N_HEADS * MAX_SEQ_LEN);
    auto att_probs = alloc_aligned<int8_t>(N_HEADS * MAX_SEQ_LEN);
    auto ffn_gate = alloc_aligned<int32_t>(HIDDEN_DIM);
    auto ffn_up = alloc_aligned<int32_t>(HIDDEN_DIM);
    auto ffn_hb = alloc_aligned<int8_t>(HIDDEN_DIM);
    auto k_cache = alloc_aligned<int8_t>(N_LAYERS * MAX_SEQ_LEN * KV_DIM);
    auto v_cache = alloc_aligned<int8_t>(N_LAYERS * MAX_SEQ_LEN * KV_DIM);
    auto cos_tab = alloc_aligned<int16_t>(MAX_SEQ_LEN * HEAD_DIM / 2);
    auto sin_tab = alloc_aligned<int16_t>(MAX_SEQ_LEN * HEAD_DIM / 2);

    // Initialize
    std::cout << "Initializing..." << std::endl;
    init_random_i16(rms_att_w, N_LAYERS * DIM);
    init_random_i16(rms_ffn_w, N_LAYERS * DIM);
    init_random_i8(wq_t, N_LAYERS * DIM * DIM);
    init_random_i8(wk_t, N_LAYERS * KV_DIM * DIM);
    init_random_i8(wv_t, N_LAYERS * KV_DIM * DIM);
    init_random_i8(wo_t, N_LAYERS * DIM * DIM);
    init_random_i8(w1_t, N_LAYERS * HIDDEN_DIM * DIM);
    init_random_i8(w2_t, N_LAYERS * DIM * HIDDEN_DIM);
    init_random_i8(w3_t, N_LAYERS * HIDDEN_DIM * DIM);
    init_random_i8(x, DIM);
    init_rope_tables(cos_tab, sin_tab, MAX_SEQ_LEN, HEAD_DIM / 2);
    memset(k_cache, 0, N_LAYERS * MAX_SEQ_LEN * KV_DIM);
    memset(v_cache, 0, N_LAYERS * MAX_SEQ_LEN * KV_DIM);

    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 3; i++) {
        forward_token_parallel(x, rms_att_w, rms_ffn_w,
            wq_t, wk_t, wv_t, wo_t, w1_t, w2_t, w3_t,
            k_cache, v_cache, cos_tab, sin_tab,
            xb_i16, xb_i8, q_i32, k_i32, v_i32,
            attn_out, att_scores, att_probs,
            ffn_gate, ffn_up, ffn_hb, i);
    }

    // Benchmark decode
    std::cout << "\n--- Parallel Decode Benchmark ---" << std::endl;
    memset(k_cache, 0, N_LAYERS * MAX_SEQ_LEN * KV_DIM);
    memset(v_cache, 0, N_LAYERS * MAX_SEQ_LEN * KV_DIM);

    int positions[] = {0, 10, 100, 500};
    for (int pos : positions) {
        // Fill cache up to pos
        for (int p = 0; p < pos; p++) {
            forward_token_parallel(x, rms_att_w, rms_ffn_w,
                wq_t, wk_t, wv_t, wo_t, w1_t, w2_t, w3_t,
                k_cache, v_cache, cos_tab, sin_tab,
                xb_i16, xb_i8, q_i32, k_i32, v_i32,
                attn_out, att_scores, att_probs,
                ffn_gate, ffn_up, ffn_hb, p);
        }

        auto start = std::chrono::high_resolution_clock::now();
        forward_token_parallel(x, rms_att_w, rms_ffn_w,
            wq_t, wk_t, wv_t, wo_t, w1_t, w2_t, w3_t,
            k_cache, v_cache, cos_tab, sin_tab,
            xb_i16, xb_i8, q_i32, k_i32, v_i32,
            attn_out, att_scores, att_probs,
            ffn_gate, ffn_up, ffn_hb, pos);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("  pos=%3d: %.2f ms (%.1f tok/s)\n", pos, ms, 1000.0 / ms);
    }

    // Prefill benchmark - this is where parallelization shines!
    std::cout << "\n--- Parallel Prefill Benchmark ---" << std::endl;
    std::cout << "+---------+----------+------------+" << std::endl;
    std::cout << "| Tokens  | Time(ms) | Tok/s      |" << std::endl;
    std::cout << "+---------+----------+------------+" << std::endl;

    int batch_sizes[] = {32, 64, 128, 256};
    for (int batch : batch_sizes) {
        memset(k_cache, 0, N_LAYERS * MAX_SEQ_LEN * KV_DIM);
        memset(v_cache, 0, N_LAYERS * MAX_SEQ_LEN * KV_DIM);

        // Allocate batch buffers
        auto x_batch = alloc_aligned<int8_t>(batch * DIM);
        auto xb_i16_batch = alloc_aligned<int16_t>(batch * DIM * 2);
        auto xb_i8_batch = alloc_aligned<int8_t>(batch * DIM);
        auto q_batch = alloc_aligned<int32_t>(batch * DIM);
        auto k_batch = alloc_aligned<int32_t>(batch * KV_DIM);
        auto v_batch = alloc_aligned<int32_t>(batch * KV_DIM);
        auto attn_out_batch = alloc_aligned<int32_t>(batch * DIM);
        auto ffn_gate_batch = alloc_aligned<int32_t>(batch * HIDDEN_DIM);
        auto ffn_up_batch = alloc_aligned<int32_t>(batch * HIDDEN_DIM);
        auto ffn_hb_batch = alloc_aligned<int8_t>(batch * HIDDEN_DIM);

        init_random_i8(x_batch, batch * DIM);

        // Warmup
        for (int layer = 0; layer < N_LAYERS; layer++) {
            // Q, K, V parallel matmuls with batch
            matmul_int8_parallel(x_batch, wq_t + layer * DIM * DIM, q_batch, batch, DIM, DIM);
            matmul_int8_parallel(x_batch, wk_t + layer * KV_DIM * DIM, k_batch, batch, KV_DIM, DIM);
            matmul_int8_parallel(x_batch, wv_t + layer * KV_DIM * DIM, v_batch, batch, KV_DIM, DIM);
            // FFN parallel matmuls
            matmul_int8_parallel(x_batch, w1_t + layer * HIDDEN_DIM * DIM, ffn_gate_batch, batch, HIDDEN_DIM, DIM);
            matmul_int8_parallel(x_batch, w3_t + layer * HIDDEN_DIM * DIM, ffn_up_batch, batch, HIDDEN_DIM, DIM);
            matmul_int8_parallel(ffn_hb_batch, w2_t + layer * DIM * HIDDEN_DIM, attn_out_batch, batch, DIM, HIDDEN_DIM);
        }

        // Benchmark - measure just the matmul-heavy forward pass
        auto start = std::chrono::high_resolution_clock::now();

        for (int layer = 0; layer < N_LAYERS; layer++) {
            // Attention matmuls (batch x DIM) @ (DIM x DIM) -> (batch x DIM)
            matmul_int8_parallel(x_batch, wq_t + layer * DIM * DIM, q_batch, batch, DIM, DIM);
            matmul_int8_parallel(x_batch, wk_t + layer * KV_DIM * DIM, k_batch, batch, KV_DIM, DIM);
            matmul_int8_parallel(x_batch, wv_t + layer * KV_DIM * DIM, v_batch, batch, KV_DIM, DIM);
            matmul_int8_parallel(xb_i8_batch, wo_t + layer * DIM * DIM, attn_out_batch, batch, DIM, DIM);

            // FFN matmuls (batch x DIM) @ (DIM x HIDDEN) -> (batch x HIDDEN)
            matmul_int8_parallel(x_batch, w1_t + layer * HIDDEN_DIM * DIM, ffn_gate_batch, batch, HIDDEN_DIM, DIM);
            matmul_int8_parallel(x_batch, w3_t + layer * HIDDEN_DIM * DIM, ffn_up_batch, batch, HIDDEN_DIM, DIM);
            // Down: (batch x HIDDEN) @ (HIDDEN x DIM) -> (batch x DIM)
            matmul_int8_parallel(ffn_hb_batch, w2_t + layer * DIM * HIDDEN_DIM, attn_out_batch, batch, DIM, HIDDEN_DIM);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double toks = batch * 1000.0 / ms;

        printf("| %5d   | %8.2f | %10.1f |\n", batch, ms, toks);

        free(x_batch); free(xb_i16_batch); free(xb_i8_batch);
        free(q_batch); free(k_batch); free(v_batch);
        free(attn_out_batch); free(ffn_gate_batch); free(ffn_up_batch); free(ffn_hb_batch);
    }
    std::cout << "+---------+----------+------------+" << std::endl;

    std::cout << "\n--- Memory ---" << std::endl;
    size_t weights_mb = (N_LAYERS * (DIM*DIM + KV_DIM*DIM*2 + DIM*DIM +
                         HIDDEN_DIM*DIM*2 + DIM*HIDDEN_DIM) + N_LAYERS*DIM*2) / (1024*1024);
    size_t kv_mb = (N_LAYERS * MAX_SEQ_LEN * KV_DIM * 2) / (1024*1024);
    printf("  Weights: ~%zu MB\n", weights_mb);
    printf("  KV Cache (max): ~%zu MB\n", kv_mb);

    return 0;
}
