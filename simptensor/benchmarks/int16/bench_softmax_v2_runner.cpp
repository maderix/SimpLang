// Runner for improved Softmax benchmarks
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <dlfcn.h>

#define MEMREF_I32 int32_t*, int32_t*, int64_t, int64_t, int64_t
#define MEMREF_I16 int16_t*, int16_t*, int64_t, int64_t, int64_t
#define PASS_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_I16(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

using softmax_fn = int32_t (*)(MEMREF_I32, MEMREF_I16, MEMREF_I32);

void softmax_fp32_ref(float* scores, float* probs, int size) {
    float max_score = scores[0];
    for (int i = 1; i < size; i++) {
        if (scores[i] > max_score) max_score = scores[i];
    }
    float sum = 0;
    for (int i = 0; i < size; i++) {
        probs[i] = expf(scores[i] - max_score);
        sum += probs[i];
    }
    for (int i = 0; i < size; i++) {
        probs[i] /= sum;
    }
}

float q15_to_float(int16_t q) { return (float)q / 32768.0f; }

void test_softmax(const char* name, softmax_fn fn, int size, int iters) {
    int32_t* scores_i32 = (int32_t*)aligned_alloc(64, size * sizeof(int32_t));
    int16_t* probs_i16 = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));
    int32_t* exp_buf = (int32_t*)aligned_alloc(64, size * sizeof(int32_t));
    float* scores_fp32 = (float*)aligned_alloc(64, size * sizeof(float));
    float* probs_fp32 = (float*)aligned_alloc(64, size * sizeof(float));

    // Initialize with realistic attention scores
    srand(42);
    for (int i = 0; i < size; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 20.0f;  // -10 to 10
        scores_fp32[i] = val;
        scores_i32[i] = (int32_t)(val * 16);  // Scale to reasonable int range
    }

    // Warmup
    for (int i = 0; i < 10; i++)
        fn(PASS_I32(scores_i32, size), PASS_I16(probs_i16, size), PASS_I32(exp_buf, size));

    // Benchmark INT
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        fn(PASS_I32(scores_i32, size), PASS_I16(probs_i16, size), PASS_I32(exp_buf, size));
    auto t1 = std::chrono::high_resolution_clock::now();
    double int_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    // Benchmark FP32
    for (int i = 0; i < 10; i++)
        softmax_fp32_ref(scores_fp32, probs_fp32, size);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        softmax_fp32_ref(scores_fp32, probs_fp32, size);
    t1 = std::chrono::high_resolution_clock::now();
    double fp32_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    // Compute error
    softmax_fp32_ref(scores_fp32, probs_fp32, size);
    fn(PASS_I32(scores_i32, size), PASS_I16(probs_i16, size), PASS_I32(exp_buf, size));

    float max_err = 0, sum_err = 0;
    for (int i = 0; i < size; i++) {
        float err = fabsf(probs_fp32[i] - q15_to_float(probs_i16[i]));
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    printf("%s (size=%d):\n", name, size);
    printf("  FP32: %.2f us (%.0f M/s)\n", fp32_us, size/fp32_us);
    printf("  INT: %.2f us (%.0f M/s)\n", int_us, size/int_us);
    printf("  Speedup: %.1fx\n", fp32_us/int_us);
    printf("  Error: max=%.4f avg=%.4f\n\n", max_err, sum_err/size);

    free(scores_i32); free(probs_i16); free(exp_buf);
    free(scores_fp32); free(probs_fp32);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <so>\n"; return 1; }

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { std::cerr << dlerror() << "\n"; return 1; }

    const int ITERS = 10000;

    std::cout << "=== Improved INT Softmax Benchmark ===\n\n";

    // Quadratic
    std::cout << "--- Quadratic exp() ---\n";
    auto q_64 = (softmax_fn)dlsym(h, "bench_softmax_v2_64");
    auto q_512 = (softmax_fn)dlsym(h, "bench_softmax_v2_512");
    auto q_2048 = (softmax_fn)dlsym(h, "bench_softmax_v2_2048");
    if (q_64) test_softmax("Quadratic", q_64, 64, ITERS);
    if (q_512) test_softmax("Quadratic", q_512, 512, ITERS);
    if (q_2048) test_softmax("Quadratic", q_2048, 2048, ITERS);

    // Cubic
    std::cout << "--- Cubic exp() ---\n";
    auto c_64 = (softmax_fn)dlsym(h, "bench_softmax_cubic_64");
    auto c_512 = (softmax_fn)dlsym(h, "bench_softmax_cubic_512");
    auto c_2048 = (softmax_fn)dlsym(h, "bench_softmax_cubic_2048");
    if (c_64) test_softmax("Cubic", c_64, 64, ITERS);
    if (c_512) test_softmax("Cubic", c_512, 512, ITERS);
    if (c_2048) test_softmax("Cubic", c_2048, 2048, ITERS);

    dlclose(h);
    return 0;
}
