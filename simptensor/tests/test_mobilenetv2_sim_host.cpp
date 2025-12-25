#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <chrono>

#define MEMREF_I8_2D(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)
#define MEMREF_I32_2D(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)
#define MEMREF_I32_1D(ptr, n) (ptr), (ptr), (int64_t)0, (int64_t)(n), (int64_t)1

// Full inverted residual (with depthwise)
typedef int32_t (*InvResFullFn)(
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // input
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // expand_filter
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // expand_scale
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // expand_bias
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // dw_filter
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // dw_scale
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // dw_bias
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // project_filter
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // project_scale
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // project_bias
    int8_t*, int8_t*, int64_t, int64_t, int64_t    // output
);

// Simplified inverted residual (no depthwise params)
typedef int32_t (*InvResSimpleFn)(
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // input
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // expand_filter
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // expand_scale
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // expand_bias
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // project_filter
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // project_scale
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // project_bias
    int8_t*, int8_t*, int64_t, int64_t, int64_t    // output
);

struct LayerConfig {
    const char* name;
    const char* symbol;
    int M;           // spatial (padded)
    int C_in;        // input channels
    int C_expanded;  // expanded channels
    int iters;
    bool has_dw;     // has depthwise params
};

double bench_invres_full(void* fn_ptr, const LayerConfig& cfg) {
    auto fn = (InvResFullFn)fn_ptr;

    int8_t* input = (int8_t*)aligned_alloc(64, cfg.M * cfg.C_in);
    int8_t* expand_f = (int8_t*)aligned_alloc(64, cfg.C_in * cfg.C_expanded);
    int32_t* expand_s = (int32_t*)aligned_alloc(64, cfg.C_expanded * 4);
    int32_t* expand_b = (int32_t*)aligned_alloc(64, cfg.C_expanded * 4);
    int8_t* dw_f = (int8_t*)aligned_alloc(64, cfg.C_expanded * cfg.C_expanded);
    int32_t* dw_s = (int32_t*)aligned_alloc(64, cfg.C_expanded * 4);
    int32_t* dw_b = (int32_t*)aligned_alloc(64, cfg.C_expanded * 4);
    int8_t* project_f = (int8_t*)aligned_alloc(64, cfg.C_expanded * cfg.C_in);
    int32_t* project_s = (int32_t*)aligned_alloc(64, cfg.C_in * 4);
    int32_t* project_b = (int32_t*)aligned_alloc(64, cfg.C_in * 4);
    int8_t* output = (int8_t*)aligned_alloc(64, cfg.M * cfg.C_in);

    memset(input, 1, cfg.M * cfg.C_in);
    memset(expand_f, 1, cfg.C_in * cfg.C_expanded);
    memset(dw_f, 1, cfg.C_expanded * cfg.C_expanded);
    memset(project_f, 1, cfg.C_expanded * cfg.C_in);
    for (int i = 0; i < cfg.C_expanded; i++) {
        expand_s[i] = 256; expand_b[i] = 0;
        dw_s[i] = 256; dw_b[i] = 0;
    }
    for (int i = 0; i < cfg.C_in; i++) { project_s[i] = 256; project_b[i] = 0; }

    // Warmup
    fn(MEMREF_I8_2D(input, cfg.M, cfg.C_in),
       MEMREF_I8_2D(expand_f, cfg.C_in, cfg.C_expanded),
       MEMREF_I32_1D(expand_s, cfg.C_expanded),
       MEMREF_I32_1D(expand_b, cfg.C_expanded),
       MEMREF_I8_2D(dw_f, cfg.C_expanded, cfg.C_expanded),
       MEMREF_I32_1D(dw_s, cfg.C_expanded),
       MEMREF_I32_1D(dw_b, cfg.C_expanded),
       MEMREF_I8_2D(project_f, cfg.C_expanded, cfg.C_in),
       MEMREF_I32_1D(project_s, cfg.C_in),
       MEMREF_I32_1D(project_b, cfg.C_in),
       MEMREF_I8_2D(output, cfg.M, cfg.C_in));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.iters; i++) {
        fn(MEMREF_I8_2D(input, cfg.M, cfg.C_in),
           MEMREF_I8_2D(expand_f, cfg.C_in, cfg.C_expanded),
           MEMREF_I32_1D(expand_s, cfg.C_expanded),
           MEMREF_I32_1D(expand_b, cfg.C_expanded),
           MEMREF_I8_2D(dw_f, cfg.C_expanded, cfg.C_expanded),
           MEMREF_I32_1D(dw_s, cfg.C_expanded),
           MEMREF_I32_1D(dw_b, cfg.C_expanded),
           MEMREF_I8_2D(project_f, cfg.C_expanded, cfg.C_in),
           MEMREF_I32_1D(project_s, cfg.C_in),
           MEMREF_I32_1D(project_b, cfg.C_in),
           MEMREF_I8_2D(output, cfg.M, cfg.C_in));
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / cfg.iters;

    free(input); free(expand_f); free(expand_s); free(expand_b);
    free(dw_f); free(dw_s); free(dw_b);
    free(project_f); free(project_s); free(project_b); free(output);

    return ms;
}

double bench_invres_simple(void* fn_ptr, const LayerConfig& cfg) {
    auto fn = (InvResSimpleFn)fn_ptr;

    int8_t* input = (int8_t*)aligned_alloc(64, cfg.M * cfg.C_in);
    int8_t* expand_f = (int8_t*)aligned_alloc(64, cfg.C_in * cfg.C_expanded);
    int32_t* expand_s = (int32_t*)aligned_alloc(64, cfg.C_expanded * 4);
    int32_t* expand_b = (int32_t*)aligned_alloc(64, cfg.C_expanded * 4);
    int8_t* project_f = (int8_t*)aligned_alloc(64, cfg.C_expanded * cfg.C_in);
    int32_t* project_s = (int32_t*)aligned_alloc(64, cfg.C_in * 4);
    int32_t* project_b = (int32_t*)aligned_alloc(64, cfg.C_in * 4);
    int8_t* output = (int8_t*)aligned_alloc(64, cfg.M * cfg.C_in);

    memset(input, 1, cfg.M * cfg.C_in);
    memset(expand_f, 1, cfg.C_in * cfg.C_expanded);
    memset(project_f, 1, cfg.C_expanded * cfg.C_in);
    for (int i = 0; i < cfg.C_expanded; i++) { expand_s[i] = 256; expand_b[i] = 0; }
    for (int i = 0; i < cfg.C_in; i++) { project_s[i] = 256; project_b[i] = 0; }

    // Warmup
    fn(MEMREF_I8_2D(input, cfg.M, cfg.C_in),
       MEMREF_I8_2D(expand_f, cfg.C_in, cfg.C_expanded),
       MEMREF_I32_1D(expand_s, cfg.C_expanded),
       MEMREF_I32_1D(expand_b, cfg.C_expanded),
       MEMREF_I8_2D(project_f, cfg.C_expanded, cfg.C_in),
       MEMREF_I32_1D(project_s, cfg.C_in),
       MEMREF_I32_1D(project_b, cfg.C_in),
       MEMREF_I8_2D(output, cfg.M, cfg.C_in));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.iters; i++) {
        fn(MEMREF_I8_2D(input, cfg.M, cfg.C_in),
           MEMREF_I8_2D(expand_f, cfg.C_in, cfg.C_expanded),
           MEMREF_I32_1D(expand_s, cfg.C_expanded),
           MEMREF_I32_1D(expand_b, cfg.C_expanded),
           MEMREF_I8_2D(project_f, cfg.C_expanded, cfg.C_in),
           MEMREF_I32_1D(project_s, cfg.C_in),
           MEMREF_I32_1D(project_b, cfg.C_in),
           MEMREF_I8_2D(output, cfg.M, cfg.C_in));
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / cfg.iters;

    free(input); free(expand_f); free(expand_s); free(expand_b);
    free(project_f); free(project_s); free(project_b); free(output);

    return ms;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <mobilenetv2_sim.so>\n", argv[0]);
        return 1;
    }

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }

    printf("=== MobileNetV2 INT8 Simulation ===\n\n");

    // MobileNetV2 layers - all use full inverted residual with dw
    LayerConfig layers[] = {
        // Stage 2: 56x56, 32ch, expand=6
        {"invres_32ch_1", "inverted_residual_32ch", 3200, 32, 192, 50, true},
        {"invres_32ch_2", "inverted_residual_32ch", 3200, 32, 192, 50, true},
        // Stage 3: 28x28, 64ch, expand=6
        {"invres_64ch_1", "inverted_residual_64ch", 832, 64, 384, 100, true},
        {"invres_64ch_2", "inverted_residual_64ch", 832, 64, 384, 100, true},
        {"invres_64ch_3", "inverted_residual_64ch", 832, 64, 384, 100, true},
        // Stage 4: 14x14, 96ch, expand=6 (no dw in kernel)
        {"invres_96ch_1", "inverted_residual_96ch", 256, 96, 576, 200, false},
        {"invres_96ch_2", "inverted_residual_96ch", 256, 96, 576, 200, false},
        {"invres_96ch_3", "inverted_residual_96ch", 256, 96, 576, 200, false},
    };

    double total_ms = 0;
    double total_ops = 0;

    printf("%-15s %20s %10s %10s\n", "Layer", "Shape", "Time(ms)", "GIOP/s");
    printf("----------------------------------------------------------\n");

    for (const auto& cfg : layers) {
        void* fn = dlsym(h, cfg.symbol);
        if (!fn) {
            printf("%-15s not found: %s\n", cfg.name, cfg.symbol);
            continue;
        }

        double ms = cfg.has_dw ? bench_invres_full(fn, cfg) : bench_invres_simple(fn, cfg);
        // Ops: expand (M*Cin*Cexp) + project (M*Cexp*Cin) = 2*M*Cin*Cexp
        double ops = 4.0 * cfg.M * cfg.C_in * cfg.C_expanded;
        double giops = ops / ms / 1e6;
        total_ms += ms;
        total_ops += ops;

        char shape[32];
        snprintf(shape, sizeof(shape), "%dx%d->%d->%d",
                 cfg.M, cfg.C_in, cfg.C_expanded, cfg.C_in);
        printf("%-15s %20s %10.3f %10.0f\n", cfg.name, shape, ms, giops);
    }

    printf("----------------------------------------------------------\n");
    printf("%-15s %20s %10.3f %10.0f\n", "TOTAL", "", total_ms, total_ops / total_ms / 1e6);

    double fps = 1000.0 / total_ms;
    printf("\nEstimated throughput: %.1f images/sec\n", fps);

    dlclose(h);
    return 0;
}
