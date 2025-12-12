#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <dlfcn.h>

#define MEMREF_I8 int8_t*, int8_t*, int64_t, int64_t, int64_t
#define MEMREF_I16 int16_t*, int16_t*, int64_t, int64_t, int64_t
#define MEMREF_I32 int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_I8(p,s) p,p,0LL,(int64_t)(s),1LL
#define PASS_I16(p,s) p,p,0LL,(int64_t)(s),1LL
#define PASS_I32(p,s) p,p,0LL,(int64_t)(s),1LL

using rope_fn = int32_t(*)(MEMREF_I32, MEMREF_I16, MEMREF_I16, MEMREF_I32);
using residual_fn = int32_t(*)(MEMREF_I32, MEMREF_I32, MEMREF_I32);
using silu_mul_fn = int32_t(*)(MEMREF_I32, MEMREF_I32, MEMREF_I8, MEMREF_I32);

// FP32 references - volatile to prevent optimization
volatile float sink = 0.0f;

__attribute__((noinline))
void ref_rope_fp32(float* __restrict__ q, float* __restrict__ cos_tab, float* __restrict__ sin_tab, float* __restrict__ out, int pairs) {
    for (int i = 0; i < pairs; i++) {
        float q0 = q[i*2], q1 = q[i*2+1];
        float c = cos_tab[i], s = sin_tab[i];
        out[i*2] = q0 * c - q1 * s;
        out[i*2+1] = q0 * s + q1 * c;
    }
    sink = out[0];
}

__attribute__((noinline))
void ref_residual_fp32(float* __restrict__ x, float* __restrict__ y, float* __restrict__ out, int n) {
    for (int i = 0; i < n; i++) out[i] = x[i] + y[i];
    sink = out[0];
}

__attribute__((noinline))
void ref_silu_mul_fp32(float* __restrict__ gate, float* __restrict__ up, float* __restrict__ out, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
    sink = out[0];
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <so>\n"; return 1; }
    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { std::cerr << dlerror() << "\n"; return 1; }

    auto rope_64 = (rope_fn)dlsym(h, "rope_i32_64");
    auto rope_2048 = (rope_fn)dlsym(h, "rope_i32_2048");
    auto residual = (residual_fn)dlsym(h, "residual_add_2048");
    auto silu_mul = (silu_mul_fn)dlsym(h, "ffn_silu_mul_8192_v2");

    const int ITERS = 10000;
    std::cout << "=== Remaining Ops Benchmark (INT8/INT16/INT32) ===\n\n";

    // ============ RoPE 64 ============
    if (rope_64) {
        int32_t* q_i32 = (int32_t*)aligned_alloc(64, 64 * sizeof(int32_t));
        int16_t* cos_i16 = (int16_t*)aligned_alloc(64, 32 * sizeof(int16_t));
        int16_t* sin_i16 = (int16_t*)aligned_alloc(64, 32 * sizeof(int16_t));
        int32_t* out_i32 = (int32_t*)aligned_alloc(64, 64 * sizeof(int32_t));
        float* q_fp32 = (float*)aligned_alloc(64, 64 * sizeof(float));
        float* cos_fp32 = (float*)aligned_alloc(64, 32 * sizeof(float));
        float* sin_fp32 = (float*)aligned_alloc(64, 32 * sizeof(float));
        float* out_fp32 = (float*)aligned_alloc(64, 64 * sizeof(float));

        srand(42);
        for (int i = 0; i < 64; i++) {
            float v = ((float)rand()/RAND_MAX - 0.5f) * 1000.0f;
            q_fp32[i] = v; q_i32[i] = (int32_t)v;
        }
        for (int i = 0; i < 32; i++) {
            float angle = (float)i * 0.1f;
            cos_fp32[i] = cosf(angle); sin_fp32[i] = sinf(angle);
            cos_i16[i] = (int16_t)(cos_fp32[i] * 32767);
            sin_i16[i] = (int16_t)(sin_fp32[i] * 32767);
        }

        for (int i = 0; i < 10; i++) rope_64(PASS_I32(q_i32,64), PASS_I16(cos_i16,32), PASS_I16(sin_i16,32), PASS_I32(out_i32,64));
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) rope_64(PASS_I32(q_i32,64), PASS_I16(cos_i16,32), PASS_I16(sin_i16,32), PASS_I32(out_i32,64));
        auto t1 = std::chrono::high_resolution_clock::now();
        double int_us = std::chrono::duration<double, std::micro>(t1-t0).count() / ITERS;

        for (int i = 0; i < 10; i++) ref_rope_fp32(q_fp32, cos_fp32, sin_fp32, out_fp32, 32);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) ref_rope_fp32(q_fp32, cos_fp32, sin_fp32, out_fp32, 32);
        t1 = std::chrono::high_resolution_clock::now();
        double fp32_us = std::chrono::duration<double, std::micro>(t1-t0).count() / ITERS;

        printf("RoPE (64 elements):\n  FP32: %.2f us, INT: %.2f us, Speedup: %.1fx\n\n", fp32_us, int_us, fp32_us/int_us);
        free(q_i32); free(cos_i16); free(sin_i16); free(out_i32);
        free(q_fp32); free(cos_fp32); free(sin_fp32); free(out_fp32);
    }

    // ============ RoPE 2048 ============
    if (rope_2048) {
        int32_t* q_i32 = (int32_t*)aligned_alloc(64, 2048 * sizeof(int32_t));
        int16_t* cos_i16 = (int16_t*)aligned_alloc(64, 1024 * sizeof(int16_t));
        int16_t* sin_i16 = (int16_t*)aligned_alloc(64, 1024 * sizeof(int16_t));
        int32_t* out_i32 = (int32_t*)aligned_alloc(64, 2048 * sizeof(int32_t));
        float* q_fp32 = (float*)aligned_alloc(64, 2048 * sizeof(float));
        float* cos_fp32 = (float*)aligned_alloc(64, 1024 * sizeof(float));
        float* sin_fp32 = (float*)aligned_alloc(64, 1024 * sizeof(float));
        float* out_fp32 = (float*)aligned_alloc(64, 2048 * sizeof(float));

        srand(42);
        for (int i = 0; i < 2048; i++) {
            float v = ((float)rand()/RAND_MAX - 0.5f) * 1000.0f;
            q_fp32[i] = v; q_i32[i] = (int32_t)v;
        }
        for (int i = 0; i < 1024; i++) {
            float angle = (float)i * 0.01f;
            cos_fp32[i] = cosf(angle); sin_fp32[i] = sinf(angle);
            cos_i16[i] = (int16_t)(cos_fp32[i] * 32767);
            sin_i16[i] = (int16_t)(sin_fp32[i] * 32767);
        }

        for (int i = 0; i < 10; i++) rope_2048(PASS_I32(q_i32,2048), PASS_I16(cos_i16,1024), PASS_I16(sin_i16,1024), PASS_I32(out_i32,2048));
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) rope_2048(PASS_I32(q_i32,2048), PASS_I16(cos_i16,1024), PASS_I16(sin_i16,1024), PASS_I32(out_i32,2048));
        auto t1 = std::chrono::high_resolution_clock::now();
        double int_us = std::chrono::duration<double, std::micro>(t1-t0).count() / ITERS;

        for (int i = 0; i < 10; i++) ref_rope_fp32(q_fp32, cos_fp32, sin_fp32, out_fp32, 1024);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) ref_rope_fp32(q_fp32, cos_fp32, sin_fp32, out_fp32, 1024);
        t1 = std::chrono::high_resolution_clock::now();
        double fp32_us = std::chrono::duration<double, std::micro>(t1-t0).count() / ITERS;

        printf("RoPE (2048 elements):\n  FP32: %.2f us, INT: %.2f us, Speedup: %.1fx\n\n", fp32_us, int_us, fp32_us/int_us);
        free(q_i32); free(cos_i16); free(sin_i16); free(out_i32);
        free(q_fp32); free(cos_fp32); free(sin_fp32); free(out_fp32);
    }

    // ============ Residual Add 2048 ============
    if (residual) {
        int32_t* x_i32 = (int32_t*)aligned_alloc(64, 2048 * sizeof(int32_t));
        int32_t* y_i32 = (int32_t*)aligned_alloc(64, 2048 * sizeof(int32_t));
        int32_t* out_i32 = (int32_t*)aligned_alloc(64, 2048 * sizeof(int32_t));
        float* x_fp32 = (float*)aligned_alloc(64, 2048 * sizeof(float));
        float* y_fp32 = (float*)aligned_alloc(64, 2048 * sizeof(float));
        float* out_fp32 = (float*)aligned_alloc(64, 2048 * sizeof(float));

        srand(42);
        for (int i = 0; i < 2048; i++) {
            x_fp32[i] = ((float)rand()/RAND_MAX - 0.5f) * 100.0f;
            y_fp32[i] = ((float)rand()/RAND_MAX - 0.5f) * 100.0f;
            x_i32[i] = (int32_t)(x_fp32[i] * 256);
            y_i32[i] = (int32_t)(y_fp32[i] * 256);
        }

        for (int i = 0; i < 10; i++) residual(PASS_I32(x_i32,2048), PASS_I32(y_i32,2048), PASS_I32(out_i32,2048));
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) residual(PASS_I32(x_i32,2048), PASS_I32(y_i32,2048), PASS_I32(out_i32,2048));
        auto t1 = std::chrono::high_resolution_clock::now();
        double int_us = std::chrono::duration<double, std::micro>(t1-t0).count() / ITERS;

        for (int i = 0; i < 10; i++) ref_residual_fp32(x_fp32, y_fp32, out_fp32, 2048);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) ref_residual_fp32(x_fp32, y_fp32, out_fp32, 2048);
        t1 = std::chrono::high_resolution_clock::now();
        double fp32_us = std::chrono::duration<double, std::micro>(t1-t0).count() / ITERS;

        printf("Residual Add (2048):\n  FP32: %.2f us, INT32: %.2f us, Speedup: %.1fx\n\n", fp32_us, int_us, fp32_us/int_us);
        free(x_i32); free(y_i32); free(out_i32);
        free(x_fp32); free(y_fp32); free(out_fp32);
    }

    // ============ FFN SiLU*Up 8192 ============
    if (silu_mul) {
        int32_t* gate_i32 = (int32_t*)aligned_alloc(64, 8192 * sizeof(int32_t));
        int32_t* up_i32 = (int32_t*)aligned_alloc(64, 8192 * sizeof(int32_t));
        int8_t* out_i8 = (int8_t*)aligned_alloc(64, 8192);
        int32_t* tmp = (int32_t*)aligned_alloc(64, 8192 * sizeof(int32_t));
        float* gate_fp32 = (float*)aligned_alloc(64, 8192 * sizeof(float));
        float* up_fp32 = (float*)aligned_alloc(64, 8192 * sizeof(float));
        float* out_fp32 = (float*)aligned_alloc(64, 8192 * sizeof(float));

        srand(42);
        for (int i = 0; i < 8192; i++) {
            gate_fp32[i] = ((float)rand()/RAND_MAX - 0.5f) * 4.0f;
            up_fp32[i] = ((float)rand()/RAND_MAX - 0.5f) * 4.0f;
            gate_i32[i] = (int32_t)(gate_fp32[i] * 16384);  // Q14
            up_i32[i] = (int32_t)(up_fp32[i] * 16384);
        }

        for (int i = 0; i < 10; i++) silu_mul(PASS_I32(gate_i32,8192), PASS_I32(up_i32,8192), PASS_I8(out_i8,8192), PASS_I32(tmp,8192));
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) silu_mul(PASS_I32(gate_i32,8192), PASS_I32(up_i32,8192), PASS_I8(out_i8,8192), PASS_I32(tmp,8192));
        auto t1 = std::chrono::high_resolution_clock::now();
        double int_us = std::chrono::duration<double, std::micro>(t1-t0).count() / ITERS;

        for (int i = 0; i < 10; i++) ref_silu_mul_fp32(gate_fp32, up_fp32, out_fp32, 8192);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) ref_silu_mul_fp32(gate_fp32, up_fp32, out_fp32, 8192);
        t1 = std::chrono::high_resolution_clock::now();
        double fp32_us = std::chrono::duration<double, std::micro>(t1-t0).count() / ITERS;

        printf("FFN SiLU*Up (8192):\n  FP32: %.2f us, INT: %.2f us, Speedup: %.1fx\n\n", fp32_us, int_us, fp32_us/int_us);
        free(gate_i32); free(up_i32); free(out_i8); free(tmp);
        free(gate_fp32); free(up_fp32); free(out_fp32);
    }

    dlclose(h);
    return 0;
}
