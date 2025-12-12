#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <dlfcn.h>

#define MEMREF_I32_PARAMS int32_t*, int32_t*, int64_t, int64_t, int64_t
#define MEMREF_I16_PARAMS int16_t*, int16_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_I32(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL
#define PASS_MEMREF_I16(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

using softmax_fn = int32_t (*)(MEMREF_I32_PARAMS, MEMREF_I16_PARAMS, MEMREF_I32_PARAMS);

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

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <so>\n"; return 1; }

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { std::cerr << dlerror() << "\n"; return 1; }

    auto fn_64 = (softmax_fn)dlsym(h, "bench_softmax_64");
    auto fn_512 = (softmax_fn)dlsym(h, "bench_softmax_512");
    auto fn_2048 = (softmax_fn)dlsym(h, "bench_softmax_2048");
    auto fn_4096 = (softmax_fn)dlsym(h, "bench_softmax_4096");

    const int ITERS = 10000;
    int sizes[] = {64, 512, 2048, 4096};
    softmax_fn fns[] = {fn_64, fn_512, fn_2048, fn_4096};

    std::cout << "=== INT32->INT16 Softmax Benchmark ===\n";
    std::cout << "Output: Q15 probabilities (0-32767)\n\n";

    for (int s = 0; s < 4; s++) {
        int size = sizes[s];
        softmax_fn fn = fns[s];
        if (!fn) continue;

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

        // Benchmark INT16
        for (int i = 0; i < 10; i++)
            fn(PASS_MEMREF_I32(scores_i32, size), PASS_MEMREF_I16(probs_i16, size), PASS_MEMREF_I32(exp_buf, size));
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            fn(PASS_MEMREF_I32(scores_i32, size), PASS_MEMREF_I16(probs_i16, size), PASS_MEMREF_I32(exp_buf, size));
        auto t1 = std::chrono::high_resolution_clock::now();
        double i16_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        // Benchmark FP32
        for (int i = 0; i < 10; i++)
            softmax_fp32_ref(scores_fp32, probs_fp32, size);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++)
            softmax_fp32_ref(scores_fp32, probs_fp32, size);
        t1 = std::chrono::high_resolution_clock::now();
        double fp32_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

        // Validate
        softmax_fp32_ref(scores_fp32, probs_fp32, size);
        fn(PASS_MEMREF_I32(scores_i32, size), PASS_MEMREF_I16(probs_i16, size), PASS_MEMREF_I32(exp_buf, size));

        float max_err = 0, sum_err = 0;
        for (int i = 0; i < size; i++) {
            float err = fabsf(probs_fp32[i] - q15_to_float(probs_i16[i]));
            max_err = std::max(max_err, err);
            sum_err += err;
        }

        printf("size=%d:\n  FP32: %.2f us (%.0f M/s)\n  INT: %.2f us (%.0f M/s)\n  Speedup: %.1fx\n  Error: max=%.4f avg=%.4f\n\n",
               size, fp32_us, size/fp32_us, i16_us, size/i16_us, fp32_us/i16_us, max_err, sum_err/size);

        free(scores_i32); free(probs_i16); free(exp_buf);
        free(scores_fp32); free(probs_fp32);
    }

    dlclose(h);
    return 0;
}
