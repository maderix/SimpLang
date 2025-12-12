#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <dlfcn.h>

// MLIR memref ABI: (base_ptr, aligned_ptr, offset, size, stride)
#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I32_PARAMS int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_I8(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_MEMREF_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

// Function types
using qk_fn = int32_t (*)(MEMREF_I8_PARAMS, MEMREF_I8_PARAMS, MEMREF_I32_PARAMS);
using av_fn = int32_t (*)(MEMREF_I8_PARAMS, MEMREF_I8_PARAMS, MEMREF_I32_PARAMS);

// FP32 reference: Q·K attention scores
void ref_attention_qk_fp32(const float* q, const float* k_cache, float* scores,
                           int head_dim, int seq_len) {
    for (int t = 0; t < seq_len; t++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q[d] * k_cache[t * head_dim + d];
        }
        scores[t] = score / sqrtf((float)head_dim);
    }
}

// FP32 reference: Attn·V weighted sum
void ref_attention_av_fp32(const float* probs, const float* v_cache, float* out,
                           int head_dim, int seq_len) {
    for (int d = 0; d < head_dim; d++) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            val += probs[t] * v_cache[t * head_dim + d];
        }
        out[d] = val;
    }
}

float int8_to_float(int8_t x) { return (float)x / 127.0f; }
int8_t float_to_int8(float x) {
    int val = (int)(x * 127.0f);
    if (val > 127) val = 127;
    if (val < -128) val = -128;
    return (int8_t)val;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <so>\n";
        return 1;
    }

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) {
        std::cerr << dlerror() << "\n";
        return 1;
    }

    // Load functions
    auto qk_128 = (qk_fn)dlsym(h, "attention_qk_128");
    auto qk_512 = (qk_fn)dlsym(h, "attention_qk_512");
    auto qk_2048 = (qk_fn)dlsym(h, "attention_qk_2048");

    auto av_128 = (av_fn)dlsym(h, "attention_av_128");
    auto av_512 = (av_fn)dlsym(h, "attention_av_512");
    auto av_2048 = (av_fn)dlsym(h, "attention_av_2048");

    const int HEAD_DIM = 64;
    const int ITERS = 10000;

    std::cout << "=== INT8 Attention Benchmark (VNNI) ===\n";
    std::cout << "head_dim=" << HEAD_DIM << ", comparing vs FP32 reference\n\n";

    // Test different seq_len values
    int seq_lens[] = {128, 512, 2048};
    qk_fn qk_fns[] = {qk_128, qk_512, qk_2048};
    av_fn av_fns[] = {av_128, av_512, av_2048};

    for (int s = 0; s < 3; s++) {
        int seq_len = seq_lens[s];
        qk_fn qk_fn_ptr = qk_fns[s];
        av_fn av_fn_ptr = av_fns[s];

        if (!qk_fn_ptr || !av_fn_ptr) {
            std::cout << "seq_len=" << seq_len << ": functions not found\n\n";
            continue;
        }

        // Allocate buffers
        int8_t* q_i8 = (int8_t*)aligned_alloc(64, HEAD_DIM * sizeof(int8_t));
        int8_t* k_cache_i8 = (int8_t*)aligned_alloc(64, seq_len * HEAD_DIM * sizeof(int8_t));
        int8_t* v_cache_i8 = (int8_t*)aligned_alloc(64, seq_len * HEAD_DIM * sizeof(int8_t));
        int32_t* scores_i32 = (int32_t*)aligned_alloc(64, seq_len * sizeof(int32_t));
        int8_t* probs_i8 = (int8_t*)aligned_alloc(64, seq_len * sizeof(int8_t));
        int32_t* out_i32 = (int32_t*)aligned_alloc(64, HEAD_DIM * sizeof(int32_t));

        float* q_fp32 = (float*)aligned_alloc(64, HEAD_DIM * sizeof(float));
        float* k_cache_fp32 = (float*)aligned_alloc(64, seq_len * HEAD_DIM * sizeof(float));
        float* v_cache_fp32 = (float*)aligned_alloc(64, seq_len * HEAD_DIM * sizeof(float));
        float* scores_fp32 = (float*)aligned_alloc(64, seq_len * sizeof(float));
        float* probs_fp32 = (float*)aligned_alloc(64, seq_len * sizeof(float));
        float* out_fp32 = (float*)aligned_alloc(64, HEAD_DIM * sizeof(float));

        // Initialize with realistic data
        srand(42);
        for (int i = 0; i < HEAD_DIM; i++) {
            float val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            q_fp32[i] = val;
            q_i8[i] = float_to_int8(val);
        }
        for (int i = 0; i < seq_len * HEAD_DIM; i++) {
            float kval = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            float vval = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            k_cache_fp32[i] = kval;
            v_cache_fp32[i] = vval;
            k_cache_i8[i] = float_to_int8(kval);
            v_cache_i8[i] = float_to_int8(vval);
        }
        // Uniform probs for testing
        for (int i = 0; i < seq_len; i++) {
            probs_i8[i] = 1;
            probs_fp32[i] = 1.0f / seq_len;
        }

        // ==================== Q·K Benchmark ====================
        for (int i = 0; i < 10; i++) {
            qk_fn_ptr(PASS_MEMREF_I8(q_i8, HEAD_DIM),
                      PASS_MEMREF_I8(k_cache_i8, seq_len * HEAD_DIM),
                      PASS_MEMREF_I32(scores_i32, seq_len));
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            qk_fn_ptr(PASS_MEMREF_I8(q_i8, HEAD_DIM),
                      PASS_MEMREF_I8(k_cache_i8, seq_len * HEAD_DIM),
                      PASS_MEMREF_I32(scores_i32, seq_len));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double qk_i8_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        for (int i = 0; i < 10; i++) {
            ref_attention_qk_fp32(q_fp32, k_cache_fp32, scores_fp32, HEAD_DIM, seq_len);
        }
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            ref_attention_qk_fp32(q_fp32, k_cache_fp32, scores_fp32, HEAD_DIM, seq_len);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double qk_fp32_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        // ==================== Attn·V Benchmark ====================
        for (int i = 0; i < 10; i++) {
            av_fn_ptr(PASS_MEMREF_I8(probs_i8, seq_len),
                      PASS_MEMREF_I8(v_cache_i8, seq_len * HEAD_DIM),
                      PASS_MEMREF_I32(out_i32, HEAD_DIM));
        }

        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            av_fn_ptr(PASS_MEMREF_I8(probs_i8, seq_len),
                      PASS_MEMREF_I8(v_cache_i8, seq_len * HEAD_DIM),
                      PASS_MEMREF_I32(out_i32, HEAD_DIM));
        }
        t1 = std::chrono::high_resolution_clock::now();
        double av_i8_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        for (int i = 0; i < 10; i++) {
            ref_attention_av_fp32(probs_fp32, v_cache_fp32, out_fp32, HEAD_DIM, seq_len);
        }
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            ref_attention_av_fp32(probs_fp32, v_cache_fp32, out_fp32, HEAD_DIM, seq_len);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double av_fp32_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        // Calculate throughput (ops = seq_len * head_dim MACs)
        double qk_ops = (double)seq_len * HEAD_DIM;
        double av_ops = (double)seq_len * HEAD_DIM;

        printf("seq_len=%d:\n", seq_len);
        printf("  Q·K scores:\n");
        printf("    FP32: %.2f us (%.1f GOPS)\n", qk_fp32_us, qk_ops / qk_fp32_us / 1e3);
        printf("    INT8: %.2f us (%.1f GOPS)\n", qk_i8_us, qk_ops / qk_i8_us / 1e3);
        printf("    Speedup: %.2fx\n", qk_fp32_us / qk_i8_us);
        printf("  Attn·V weighted sum:\n");
        printf("    FP32: %.2f us (%.1f GOPS)\n", av_fp32_us, av_ops / av_fp32_us / 1e3);
        printf("    INT8: %.2f us (%.1f GOPS)\n", av_i8_us, av_ops / av_i8_us / 1e3);
        printf("    Speedup: %.2fx\n", av_fp32_us / av_i8_us);
        printf("\n");

        free(q_i8); free(k_cache_i8); free(v_cache_i8);
        free(scores_i32); free(probs_i8); free(out_i32);
        free(q_fp32); free(k_cache_fp32); free(v_cache_fp32);
        free(scores_fp32); free(probs_fp32); free(out_fp32);
    }

    dlclose(h);
    return 0;
}
