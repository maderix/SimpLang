#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <dlfcn.h>

#define MEMREF_I16_PARAMS int16_t*, int16_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_I16(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

using silu_fn = int32_t (*)(MEMREF_I16_PARAMS, MEMREF_I16_PARAMS);

void silu_fp32_ref(float* x, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

int16_t float_to_q8(float f) {
    int32_t v = (int32_t)(f * 256.0f);
    return (int16_t)std::max(-32768, std::min(32767, v));
}

float q8_to_float(int16_t q) { return (float)q / 256.0f; }

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <so>\n"; return 1; }

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { std::cerr << dlerror() << "\n"; return 1; }

    auto fn_2048 = (silu_fn)dlsym(h, "bench_silu_2048");
    auto fn_8192 = (silu_fn)dlsym(h, "bench_silu_8192");
    auto fn_11008 = (silu_fn)dlsym(h, "bench_silu_11008");

    const int ITERS = 10000;
    int dims[] = {2048, 8192, 11008};
    silu_fn fns[] = {fn_2048, fn_8192, fn_11008};

    std::cout << "=== INT16 SiLU (Cubic) Benchmark ===\n\n";

    for (int d = 0; d < 3; d++) {
        int dim = dims[d];
        silu_fn fn = fns[d];
        if (!fn) continue;

        int16_t* x_i16 = (int16_t*)aligned_alloc(64, dim * sizeof(int16_t));
        int16_t* out_i16 = (int16_t*)aligned_alloc(64, dim * sizeof(int16_t));
        float* x_fp32 = (float*)aligned_alloc(64, dim * sizeof(float));
        float* out_fp32 = (float*)aligned_alloc(64, dim * sizeof(float));

        srand(42);
        for (int i = 0; i < dim; i++) {
            float val = ((float)rand() / RAND_MAX - 0.5f) * 8.0f;
            x_fp32[i] = val; x_i16[i] = float_to_q8(val);
        }

        for (int i = 0; i < 10; i++) fn(PASS_MEMREF_I16(x_i16, dim), PASS_MEMREF_I16(out_i16, dim));
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) fn(PASS_MEMREF_I16(x_i16, dim), PASS_MEMREF_I16(out_i16, dim));
        auto t1 = std::chrono::high_resolution_clock::now();
        double i16_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        for (int i = 0; i < 10; i++) silu_fp32_ref(x_fp32, out_fp32, dim);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) silu_fp32_ref(x_fp32, out_fp32, dim);
        t1 = std::chrono::high_resolution_clock::now();
        double fp32_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        silu_fp32_ref(x_fp32, out_fp32, dim);
        fn(PASS_MEMREF_I16(x_i16, dim), PASS_MEMREF_I16(out_i16, dim));

        float max_err = 0, sum_err = 0;
        for (int i = 0; i < dim; i++) {
            float err = fabsf(out_fp32[i] - q8_to_float(out_i16[i]));
            max_err = std::max(max_err, err); sum_err += err;
        }

        printf("dim=%d:\n  FP32: %.2f us (%.0f M/s)\n  INT16: %.2f us (%.0f M/s)\n  Speedup: %.1fx\n  Error: max=%.4f avg=%.4f\n\n",
               dim, fp32_us, dim/fp32_us, i16_us, dim/i16_us, fp32_us/i16_us, max_err, sum_err/dim);

        free(x_i16); free(out_i16); free(x_fp32); free(out_fp32);
    }
    dlclose(h);
    return 0;
}
