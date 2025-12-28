#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <chrono>

#define MEMREF_I8_2D(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)
#define MEMREF_I32_2D(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)
#define MEMREF_I8_1D(ptr, n) (ptr), (ptr), (int64_t)0, (int64_t)(n), (int64_t)1
#define MEMREF_I32_1D(ptr, n) (ptr), (ptr), (int64_t)0, (int64_t)(n), (int64_t)1

// Block function signature
typedef int32_t (*BlockFn)(
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // im2col
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // filter1
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // filter2
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // scale1
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // bias1
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // scale2
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // bias2
    int8_t*, int8_t*, int64_t, int64_t, int64_t,   // residual
    int8_t*, int8_t*, int64_t, int64_t, int64_t    // output
);

struct LayerConfig {
    const char* name;
    const char* symbol;
    int M;      // patches (padded)
    int K;      // 3x3 * channels
    int N;      // output channels
    int iters;
};

double bench_block(BlockFn fn, const LayerConfig& cfg) {
    size_t im2col_sz = (size_t)cfg.M * cfg.K;
    size_t filter_sz = (size_t)cfg.K * cfg.N;
    size_t out_sz = (size_t)cfg.M * cfg.N;

    int8_t* im2col = (int8_t*)aligned_alloc(64, im2col_sz);
    int8_t* filter1 = (int8_t*)aligned_alloc(64, filter_sz);
    int8_t* filter2 = (int8_t*)aligned_alloc(64, filter_sz);
    int32_t* scale1 = (int32_t*)aligned_alloc(64, cfg.N * 4);
    int32_t* bias1 = (int32_t*)aligned_alloc(64, cfg.N * 4);
    int32_t* scale2 = (int32_t*)aligned_alloc(64, cfg.N * 4);
    int32_t* bias2 = (int32_t*)aligned_alloc(64, cfg.N * 4);
    int8_t* residual = (int8_t*)aligned_alloc(64, out_sz);
    int8_t* output = (int8_t*)aligned_alloc(64, out_sz);

    // Initialize
    memset(im2col, 1, im2col_sz);
    memset(filter1, 1, filter_sz);
    memset(filter2, 1, filter_sz);
    for (int i = 0; i < cfg.N; i++) {
        scale1[i] = 256; bias1[i] = 0;
        scale2[i] = 256; bias2[i] = 0;
    }
    memset(residual, 1, out_sz);
    memset(output, 0, out_sz);

    // Warmup
    fn(MEMREF_I8_2D(im2col, cfg.M, cfg.K),
       MEMREF_I8_2D(filter1, cfg.K, cfg.N),
       MEMREF_I8_2D(filter2, cfg.K, cfg.N),
       MEMREF_I32_1D(scale1, cfg.N),
       MEMREF_I32_1D(bias1, cfg.N),
       MEMREF_I32_1D(scale2, cfg.N),
       MEMREF_I32_1D(bias2, cfg.N),
       MEMREF_I8_2D(residual, cfg.M, cfg.N),
       MEMREF_I8_2D(output, cfg.M, cfg.N));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.iters; i++) {
        fn(MEMREF_I8_2D(im2col, cfg.M, cfg.K),
           MEMREF_I8_2D(filter1, cfg.K, cfg.N),
           MEMREF_I8_2D(filter2, cfg.K, cfg.N),
           MEMREF_I32_1D(scale1, cfg.N),
           MEMREF_I32_1D(bias1, cfg.N),
           MEMREF_I32_1D(scale2, cfg.N),
           MEMREF_I32_1D(bias2, cfg.N),
           MEMREF_I8_2D(residual, cfg.M, cfg.N),
           MEMREF_I8_2D(output, cfg.M, cfg.N));
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / cfg.iters;

    // 2 convs per block: 2 * (2*M*K*N)
    double ops = 4.0 * cfg.M * cfg.K * cfg.N;
    double giops = ops / ms / 1e6;

    free(im2col); free(filter1); free(filter2);
    free(scale1); free(bias1); free(scale2); free(bias2);
    free(residual); free(output);

    return ms;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <resnet18_sim.so>\n", argv[0]);
        return 1;
    }

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }

    printf("=== ResNet18 INT8 Simulation ===\n\n");

    // ResNet18 has 2 blocks per layer
    LayerConfig layers[] = {
        // Layer 1: 56x56, 64ch (2 blocks)
        {"layer1_block1", "layer1_block", 3200, 576, 64, 50},
        {"layer1_block2", "layer1_block", 3200, 576, 64, 50},
        // Layer 2: 28x28, 128ch (2 blocks)
        {"layer2_block1", "layer2_block", 832, 1152, 128, 100},
        {"layer2_block2", "layer2_block", 832, 1152, 128, 100},
        // Layer 3: 14x14, 256ch (2 blocks)
        {"layer3_block1", "layer3_block", 256, 2304, 256, 200},
        {"layer3_block2", "layer3_block", 256, 2304, 256, 200},
        // Layer 4: 7x7, 512ch (2 blocks)
        {"layer4_block1", "layer4_block", 64, 4608, 512, 500},
        {"layer4_block2", "layer4_block", 64, 4608, 512, 500},
    };

    double total_ms = 0;
    double total_ops = 0;

    printf("%-15s %15s %12s %10s\n", "Layer", "Shape (MxKxN)", "Time (ms)", "GIOP/s");
    printf("--------------------------------------------------------------\n");

    for (const auto& cfg : layers) {
        BlockFn fn = (BlockFn)dlsym(h, cfg.symbol);
        if (!fn) {
            printf("%-15s not found\n", cfg.name);
            continue;
        }

        double ms = bench_block(fn, cfg);
        double ops = 4.0 * cfg.M * cfg.K * cfg.N;
        double giops = ops / ms / 1e6;
        total_ms += ms;
        total_ops += ops;

        char shape[32];
        snprintf(shape, sizeof(shape), "%dx%dx%d", cfg.M, cfg.K, cfg.N);
        printf("%-15s %15s %12.3f %10.0f\n", cfg.name, shape, ms, giops);
    }

    printf("--------------------------------------------------------------\n");
    printf("%-15s %15s %12.3f %10.0f\n", "TOTAL", "", total_ms, total_ops / total_ms / 1e6);

    // Estimate inference throughput
    double fps = 1000.0 / total_ms;
    printf("\nEstimated throughput: %.1f images/sec\n", fps);

    dlclose(h);
    return 0;
}
