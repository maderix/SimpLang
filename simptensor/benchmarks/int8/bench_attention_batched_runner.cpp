#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <dlfcn.h>

#define MEMREF_I8_PARAMS int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I32_PARAMS int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_I8(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_MEMREF_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

using fn_t = int32_t (*)(MEMREF_I8_PARAMS, MEMREF_I8_PARAMS, MEMREF_I32_PARAMS);

// FP32 reference for 4 heads
void ref_qk_4h_fp32(const float* q, const float* k, float* scores, int seq_len) {
    for (int h = 0; h < 4; h++) {
        for (int t = 0; t < seq_len; t++) {
            float score = 0.0f;
            for (int d = 0; d < 64; d++) {
                score += q[h * 64 + d] * k[t * 64 + d];
            }
            scores[h * seq_len + t] = score / 8.0f;
        }
    }
}

void ref_av_4h_fp32(const float* probs, const float* v, float* out, int seq_len) {
    for (int h = 0; h < 4; h++) {
        for (int d = 0; d < 64; d++) {
            float val = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                val += probs[h * seq_len + t] * v[t * 64 + d];
            }
            out[h * 64 + d] = val;
        }
    }
}

int8_t f2i8(float x) {
    int v = (int)(x * 127.0f);
    return (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <so>\n"; return 1; }
    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { std::cerr << dlerror() << "\n"; return 1; }

    auto qk_4h_128 = (fn_t)dlsym(h, "attention_qk_4h_128");
    auto qk_4h_512 = (fn_t)dlsym(h, "attention_qk_4h_512");
    auto qk_4h_2048 = (fn_t)dlsym(h, "attention_qk_4h_2048");
    auto av_4h_128 = (fn_t)dlsym(h, "attention_av_4h_128");
    auto av_4h_512 = (fn_t)dlsym(h, "attention_av_4h_512");
    auto av_4h_2048 = (fn_t)dlsym(h, "attention_av_4h_2048");

    const int ITERS = 10000;
    std::cout << "=== INT8 Batched Attention (4 heads, I=4 tiling) ===\n\n";

    int seq_lens[] = {128, 512, 2048};
    fn_t qk_fns[] = {qk_4h_128, qk_4h_512, qk_4h_2048};
    fn_t av_fns[] = {av_4h_128, av_4h_512, av_4h_2048};

    for (int s = 0; s < 3; s++) {
        int seq_len = seq_lens[s];
        if (!qk_fns[s] || !av_fns[s]) { printf("seq_len=%d: not found\n", seq_len); continue; }

        int8_t* q_i8 = (int8_t*)aligned_alloc(64, 4 * 64);
        int8_t* k_i8 = (int8_t*)aligned_alloc(64, seq_len * 64);
        int8_t* v_i8 = (int8_t*)aligned_alloc(64, seq_len * 64);
        int32_t* scores_i32 = (int32_t*)aligned_alloc(64, 4 * seq_len * sizeof(int32_t));
        int8_t* probs_i8 = (int8_t*)aligned_alloc(64, 4 * seq_len);
        int32_t* out_i32 = (int32_t*)aligned_alloc(64, 4 * 64 * sizeof(int32_t));

        float* q_fp32 = (float*)aligned_alloc(64, 4 * 64 * sizeof(float));
        float* k_fp32 = (float*)aligned_alloc(64, seq_len * 64 * sizeof(float));
        float* v_fp32 = (float*)aligned_alloc(64, seq_len * 64 * sizeof(float));
        float* scores_fp32 = (float*)aligned_alloc(64, 4 * seq_len * sizeof(float));
        float* probs_fp32 = (float*)aligned_alloc(64, 4 * seq_len * sizeof(float));
        float* out_fp32 = (float*)aligned_alloc(64, 4 * 64 * sizeof(float));

        srand(42);
        for (int i = 0; i < 4 * 64; i++) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            q_fp32[i] = v; q_i8[i] = f2i8(v);
        }
        for (int i = 0; i < seq_len * 64; i++) {
            float kv = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            float vv = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            k_fp32[i] = kv; k_i8[i] = f2i8(kv);
            v_fp32[i] = vv; v_i8[i] = f2i8(vv);
        }
        for (int i = 0; i < 4 * seq_len; i++) {
            probs_i8[i] = 1;
            probs_fp32[i] = 1.0f / seq_len;
        }

        // Q·K benchmark
        for (int i = 0; i < 10; i++)
            qk_fns[s](PASS_MEMREF_I8(q_i8, 4*64), PASS_MEMREF_I8(k_i8, seq_len*64), PASS_MEMREF_I32(scores_i32, 4*seq_len));
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            qk_fns[s](PASS_MEMREF_I8(q_i8, 4*64), PASS_MEMREF_I8(k_i8, seq_len*64), PASS_MEMREF_I32(scores_i32, 4*seq_len));
        auto t1 = std::chrono::high_resolution_clock::now();
        double qk_i8_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        for (int i = 0; i < 10; i++) ref_qk_4h_fp32(q_fp32, k_fp32, scores_fp32, seq_len);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) ref_qk_4h_fp32(q_fp32, k_fp32, scores_fp32, seq_len);
        t1 = std::chrono::high_resolution_clock::now();
        double qk_fp32_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        // Attn·V benchmark
        for (int i = 0; i < 10; i++)
            av_fns[s](PASS_MEMREF_I8(probs_i8, 4*seq_len), PASS_MEMREF_I8(v_i8, seq_len*64), PASS_MEMREF_I32(out_i32, 4*64));
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            av_fns[s](PASS_MEMREF_I8(probs_i8, 4*seq_len), PASS_MEMREF_I8(v_i8, seq_len*64), PASS_MEMREF_I32(out_i32, 4*64));
        t1 = std::chrono::high_resolution_clock::now();
        double av_i8_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        for (int i = 0; i < 10; i++) ref_av_4h_fp32(probs_fp32, v_fp32, out_fp32, seq_len);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) ref_av_4h_fp32(probs_fp32, v_fp32, out_fp32, seq_len);
        t1 = std::chrono::high_resolution_clock::now();
        double av_fp32_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        double ops = 4.0 * seq_len * 64;
        printf("seq_len=%d (4 heads batched):\n", seq_len);
        printf("  Q·K: FP32=%.2fus INT8=%.2fus Speedup=%.1fx (%.0f GOPS)\n",
               qk_fp32_us, qk_i8_us, qk_fp32_us/qk_i8_us, ops/qk_i8_us/1e3);
        printf("  A·V: FP32=%.2fus INT8=%.2fus Speedup=%.1fx (%.0f GOPS)\n\n",
               av_fp32_us, av_i8_us, av_fp32_us/av_i8_us, ops/av_i8_us/1e3);

        free(q_i8); free(k_i8); free(v_i8); free(scores_i32); free(probs_i8); free(out_i32);
        free(q_fp32); free(k_fp32); free(v_fp32); free(scores_fp32); free(probs_fp32); free(out_fp32);
    }
    dlclose(h);
    return 0;
}
