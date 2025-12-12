#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <dlfcn.h>

// MLIR MemRef ABI: base_ptr, aligned_ptr, offset, size, stride
#define MEMREF_I16_PARAMS int16_t*, int16_t*, int64_t, int64_t, int64_t
#define PASS_MEMREF_I16(ptr, size) ptr, ptr, 0LL, (int64_t)(size), 1LL

// Function signatures matching MLIR ABI
using rmsnorm_fn = int32_t (*)(
    MEMREF_I16_PARAMS,  // x
    MEMREF_I16_PARAMS,  // weight
    MEMREF_I16_PARAMS   // out
);

// FP32 reference
void rmsnorm_fp32_ref(float* x, float* weight, float* out, int dim) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / dim;
    float rsqrt_ss = 1.0f / sqrtf(ss + 1e-5f);
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * rsqrt_ss * weight[i];
    }
}

// Q8 helpers
int16_t float_to_q8(float f) {
    int32_t v = (int32_t)(f * 256.0f);
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

float q8_to_float(int16_t q) {
    return (float)q / 256.0f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_so>" << std::endl;
        return 1;
    }

    void* handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        std::cerr << "Error loading: " << dlerror() << std::endl;
        return 1;
    }

    auto fn_2048 = (rmsnorm_fn)dlsym(handle, "bench_rmsnorm_2048");
    auto fn_4096 = (rmsnorm_fn)dlsym(handle, "bench_rmsnorm_4096");
    auto fn_8192 = (rmsnorm_fn)dlsym(handle, "bench_rmsnorm_8192");

    if (!fn_2048 || !fn_4096 || !fn_8192) {
        std::cerr << "Error: Could not find benchmark functions" << std::endl;
        dlclose(handle);
        return 1;
    }

    const int ITERS = 10000;
    int dims[] = {2048, 4096, 8192};
    rmsnorm_fn fns[] = {fn_2048, fn_4096, fn_8192};

    std::cout << "=== INT16 RMSNorm Benchmark ===" << std::endl;
    std::cout << "Format: Q8 (8 fractional bits, 256 = 1.0)" << std::endl;
    std::cout << "Iterations: " << ITERS << std::endl << std::endl;

    for (int d = 0; d < 3; d++) {
        int dim = dims[d];
        rmsnorm_fn fn = fns[d];

        // Allocate aligned buffers
        int16_t* x_i16 = (int16_t*)aligned_alloc(64, dim * sizeof(int16_t));
        int16_t* w_i16 = (int16_t*)aligned_alloc(64, dim * sizeof(int16_t));
        int16_t* out_i16 = (int16_t*)aligned_alloc(64, dim * sizeof(int16_t));

        float* x_fp32 = (float*)aligned_alloc(64, dim * sizeof(float));
        float* w_fp32 = (float*)aligned_alloc(64, dim * sizeof(float));
        float* out_fp32 = (float*)aligned_alloc(64, dim * sizeof(float));

        // Initialize
        srand(42);
        for (int i = 0; i < dim; i++) {
            float val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            x_fp32[i] = val;
            x_i16[i] = float_to_q8(val);
            w_fp32[i] = 1.0f;
            w_i16[i] = 256;  // 1.0 in Q8
        }
        memset(out_i16, 0, dim * sizeof(int16_t));
        memset(out_fp32, 0, dim * sizeof(float));

        // Warmup INT16
        for (int i = 0; i < 10; i++) {
            fn(PASS_MEMREF_I16(x_i16, dim),
               PASS_MEMREF_I16(w_i16, dim),
               PASS_MEMREF_I16(out_i16, dim));
        }

        // Benchmark INT16
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            fn(PASS_MEMREF_I16(x_i16, dim),
               PASS_MEMREF_I16(w_i16, dim),
               PASS_MEMREF_I16(out_i16, dim));
        }
        auto end = std::chrono::high_resolution_clock::now();
        double i16_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;

        // Warmup FP32
        for (int i = 0; i < 10; i++) {
            rmsnorm_fp32_ref(x_fp32, w_fp32, out_fp32, dim);
        }

        // Benchmark FP32
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            rmsnorm_fp32_ref(x_fp32, w_fp32, out_fp32, dim);
        }
        end = std::chrono::high_resolution_clock::now();
        double fp32_us = std::chrono::duration<double, std::micro>(end - start).count() / ITERS;

        // Validate accuracy
        rmsnorm_fp32_ref(x_fp32, w_fp32, out_fp32, dim);
        fn(PASS_MEMREF_I16(x_i16, dim),
           PASS_MEMREF_I16(w_i16, dim),
           PASS_MEMREF_I16(out_i16, dim));

        float max_err = 0.0f, sum_err = 0.0f;
        for (int i = 0; i < dim; i++) {
            float ref = out_fp32[i];
            float test = q8_to_float(out_i16[i]);
            float err = fabsf(ref - test);
            max_err = std::max(max_err, err);
            sum_err += err;
        }

        double i16_throughput = dim / i16_us;  // M elem/s
        double fp32_throughput = dim / fp32_us;

        std::cout << "dim=" << dim << ":" << std::endl;
        std::cout << "  FP32:  " << fp32_us << " us  (" << fp32_throughput << " M elem/s)" << std::endl;
        std::cout << "  INT16: " << i16_us << " us  (" << i16_throughput << " M elem/s)" << std::endl;
        std::cout << "  Speedup: " << fp32_us / i16_us << "x" << std::endl;
        std::cout << "  Max error: " << max_err << ", Avg error: " << sum_err/dim << std::endl;
        std::cout << std::endl;

        free(x_i16); free(w_i16); free(out_i16);
        free(x_fp32); free(w_fp32); free(out_fp32);
    }

    dlclose(handle);
    return 0;
}
