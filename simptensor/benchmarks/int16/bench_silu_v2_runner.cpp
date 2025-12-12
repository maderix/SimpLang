// Runner for improved SiLU benchmarks
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <dlfcn.h>

#define MEMREF_I16 int16_t*, int16_t*, int64_t, int64_t, int64_t
#define MEMREF_I32 int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_I16(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

using silu_fn = int32_t (*)(MEMREF_I16, MEMREF_I16, MEMREF_I32);

float silu_fp32(float x) {
    return x / (1.0f + expf(-x));
}

float q8_to_float(int16_t q) { return (float)q / 256.0f; }
int16_t float_to_q8(float f) {
    int v = (int)(f * 256.0f);
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

void test_silu(const char* name, silu_fn fn, int size, int iters) {
    int16_t* x = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));
    int16_t* out = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));
    int32_t* tmp = (int32_t*)aligned_alloc(64, size * sizeof(int32_t));
    float* x_fp32 = (float*)aligned_alloc(64, size * sizeof(float));
    float* out_fp32 = (float*)aligned_alloc(64, size * sizeof(float));

    // Initialize with range [-5, 5] (covers saturation regions)
    srand(42);
    for (int i = 0; i < size; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;  // [-5, 5]
        x_fp32[i] = val;
        x[i] = float_to_q8(val);
    }

    // Warmup
    for (int i = 0; i < 10; i++)
        fn(PASS_I16(x, size), PASS_I16(out, size), PASS_I32(tmp, size));

    // Benchmark INT
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        fn(PASS_I16(x, size), PASS_I16(out, size), PASS_I32(tmp, size));
    auto t1 = std::chrono::high_resolution_clock::now();
    double int_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    // Benchmark FP32
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < size; j++) out_fp32[j] = silu_fp32(x_fp32[j]);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        for (int j = 0; j < size; j++) out_fp32[j] = silu_fp32(x_fp32[j]);
    t1 = std::chrono::high_resolution_clock::now();
    double fp32_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    // Compute error
    fn(PASS_I16(x, size), PASS_I16(out, size), PASS_I32(tmp, size));
    float max_err = 0, sum_err = 0;
    for (int i = 0; i < size; i++) {
        float ref = silu_fp32(x_fp32[i]);
        float approx = q8_to_float(out[i]);
        float err = fabsf(ref - approx);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    printf("%s (dim=%d):\n", name, size);
    printf("  FP32: %.2f us (%.0f M/s)\n", fp32_us, size/fp32_us);
    printf("  INT16: %.2f us (%.0f M/s)\n", int_us, size/int_us);
    printf("  Speedup: %.1fx\n", fp32_us/int_us);
    printf("  Error: max=%.4f avg=%.4f\n\n", max_err, sum_err/size);

    free(x); free(out); free(tmp); free(x_fp32); free(out_fp32);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <so>\n"; return 1; }

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { std::cerr << dlerror() << "\n"; return 1; }

    const int ITERS = 10000;

    std::cout << "=== Improved INT16 SiLU Benchmark ===\n\n";

    // Piecewise
    auto pw_2048 = (silu_fn)dlsym(h, "bench_silu_v2_2048");
    auto pw_8192 = (silu_fn)dlsym(h, "bench_silu_v2_8192");
    if (pw_2048) test_silu("Piecewise", pw_2048, 2048, ITERS);
    if (pw_8192) test_silu("Piecewise", pw_8192, 8192, ITERS);

    // Cubic
    auto cubic_2048 = (silu_fn)dlsym(h, "bench_silu_cubic_2048");
    auto cubic_8192 = (silu_fn)dlsym(h, "bench_silu_cubic_8192");
    if (cubic_2048) test_silu("Cubic", cubic_2048, 2048, ITERS);
    if (cubic_8192) test_silu("Cubic", cubic_8192, 8192, ITERS);

    dlclose(h);
    return 0;
}
