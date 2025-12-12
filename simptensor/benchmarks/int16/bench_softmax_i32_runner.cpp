// Runner for INT32 output Softmax benchmarks
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <dlfcn.h>

#define MEMREF_I32 int32_t*, int32_t*, int64_t, int64_t, int64_t
#define PASS_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

using softmax_fn = int32_t (*)(MEMREF_I32, MEMREF_I32, MEMREF_I32);

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

void test_softmax(const char* name, softmax_fn fn, int size, int iters, float scale) {
    int32_t* scores_i32 = (int32_t*)aligned_alloc(64, size * sizeof(int32_t));
    int32_t* probs_i32 = (int32_t*)aligned_alloc(64, size * sizeof(int32_t));
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
        fn(PASS_I32(scores_i32, size), PASS_I32(probs_i32, size), PASS_I32(exp_buf, size));

    // Benchmark INT
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        fn(PASS_I32(scores_i32, size), PASS_I32(probs_i32, size), PASS_I32(exp_buf, size));
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
    fn(PASS_I32(scores_i32, size), PASS_I32(probs_i32, size), PASS_I32(exp_buf, size));

    float max_err = 0, sum_err = 0;
    for (int i = 0; i < size; i++) {
        float approx = (float)probs_i32[i] / scale;
        float err = fabsf(probs_fp32[i] - approx);
        max_err = std::max(max_err, err);
        sum_err += err;
    }

    // Check sum of probabilities
    float prob_sum = 0;
    for (int i = 0; i < size; i++) {
        prob_sum += (float)probs_i32[i] / scale;
    }

    printf("%s (size=%d, scale=%.0f):\n", name, size, scale);
    printf("  FP32: %.2f us (%.0f M/s)\n", fp32_us, size/fp32_us);
    printf("  INT32: %.2f us (%.0f M/s)\n", int_us, size/int_us);
    printf("  Speedup: %.1fx\n", fp32_us/int_us);
    printf("  Error: max=%.6f avg=%.6f\n", max_err, sum_err/size);
    printf("  Prob sum: %.6f (should be 1.0)\n\n", prob_sum);

    free(scores_i32); free(probs_i32); free(exp_buf);
    free(scores_fp32); free(probs_fp32);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <so>\n"; return 1; }

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { std::cerr << dlerror() << "\n"; return 1; }

    const int ITERS = 10000;

    std::cout << "=== INT32 Output Softmax Benchmark ===\n\n";

    // Q16 (65536 = 1.0)
    std::cout << "--- Q16 (65536 = 1.0) ---\n";
    auto q16_64 = (softmax_fn)dlsym(h, "bench_softmax_i32_64");
    auto q16_512 = (softmax_fn)dlsym(h, "bench_softmax_i32_512");
    auto q16_2048 = (softmax_fn)dlsym(h, "bench_softmax_i32_2048");
    if (q16_64) test_softmax("Q16", q16_64, 64, ITERS, 65536.0f);
    if (q16_512) test_softmax("Q16", q16_512, 512, ITERS, 65536.0f);
    if (q16_2048) test_softmax("Q16", q16_2048, 2048, ITERS, 65536.0f);

    // Q20 (1048576 = 1.0)
    std::cout << "--- Q20 (1048576 = 1.0) ---\n";
    auto q20_64 = (softmax_fn)dlsym(h, "bench_softmax_q20_64");
    auto q20_512 = (softmax_fn)dlsym(h, "bench_softmax_q20_512");
    auto q20_2048 = (softmax_fn)dlsym(h, "bench_softmax_q20_2048");
    if (q20_64) test_softmax("Q20", q20_64, 64, ITERS, 1048576.0f);
    if (q20_512) test_softmax("Q20", q20_512, 512, ITERS, 1048576.0f);
    if (q20_2048) test_softmax("Q20", q20_2048, 2048, ITERS, 1048576.0f);

    dlclose(h);
    return 0;
}
