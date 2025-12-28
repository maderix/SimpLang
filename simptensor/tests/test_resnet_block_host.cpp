#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <chrono>
#include <cmath>

// 2D memref format: base, aligned, offset, rows, cols
#define MEMREF_I8(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)
#define MEMREF_I32(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)
// 1D memref
#define MEMREF_1D_I32(ptr, n) (ptr), (ptr), (int64_t)0, (int64_t)(n), (int64_t)1

// Function signatures
typedef int32_t (*Conv3x3ReluFn)(
    int8_t*, int8_t*, int64_t, int64_t, int64_t,  // im2col
    int8_t*, int8_t*, int64_t, int64_t, int64_t,  // filter
    int8_t*, int8_t*, int64_t, int64_t, int64_t,  // output
    int32_t*, int32_t*, int64_t, int64_t, int64_t,  // scale
    int32_t*, int32_t*, int64_t, int64_t, int64_t   // bias
);

typedef int32_t (*Conv3x3LinearFn)(
    int8_t*, int8_t*, int64_t, int64_t, int64_t,  // im2col
    int8_t*, int8_t*, int64_t, int64_t, int64_t,  // filter
    int32_t*, int32_t*, int64_t, int64_t, int64_t, // output (i32)
    int32_t*, int32_t*, int64_t, int64_t, int64_t,  // scale
    int32_t*, int32_t*, int64_t, int64_t, int64_t   // bias
);

typedef int32_t (*ResidualAddReluFn)(
    int32_t*, int32_t*, int64_t, int64_t, int64_t,  // conv_out
    int8_t*, int8_t*, int64_t, int64_t, int64_t,    // residual
    int8_t*, int8_t*, int64_t, int64_t, int64_t     // output
);

// im2col for 3x3 conv with padding=1, stride=1
void im2col_3x3(const int8_t* input, int8_t* col, int H, int W, int C) {
    int out_H = H, out_W = W;
    int patch_size = 9 * C;

    for (int oh = 0; oh < out_H; oh++) {
        for (int ow = 0; ow < out_W; ow++) {
            int patch_idx = oh * out_W + ow;
            int8_t* patch = col + patch_idx * patch_size;

            int col_offset = 0;
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int ih = oh + kh - 1;  // padding=1
                    int iw = ow + kw - 1;

                    for (int c = 0; c < C; c++) {
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            patch[col_offset] = input[(ih * W + iw) * C + c];
                        } else {
                            patch[col_offset] = 0;  // zero padding
                        }
                        col_offset++;
                    }
                }
            }
        }
    }
}

// Reference implementation for verification
void conv3x3_relu_ref(const int8_t* im2col, const int8_t* filter,
                      const int32_t* scale, const int32_t* bias,
                      int8_t* output, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)im2col[i * K + k] * (int32_t)filter[k * N + j];
            }
            int32_t scaled = (acc * scale[j]) >> 16;
            int32_t biased = scaled + bias[j];
            if (biased < 0) biased = 0;
            if (biased > 127) biased = 127;
            output[i * N + j] = (int8_t)biased;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <resnet_block.so>\n", argv[0]);
        return 1;
    }

    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }

    // Load functions
    Conv3x3ReluFn conv3x3_relu = (Conv3x3ReluFn)dlsym(h, "conv3x3_relu");
    Conv3x3LinearFn conv3x3_linear = (Conv3x3LinearFn)dlsym(h, "conv3x3_linear");
    ResidualAddReluFn residual_add_relu = (ResidualAddReluFn)dlsym(h, "residual_add_relu");

    if (!conv3x3_relu) {
        printf("conv3x3_relu not found\n");
        return 1;
    }

    printf("=== ResNet Block INT8 Test ===\n\n");

    // Dimensions: 32x32 spatial, 64 channels
    const int H = 32, W = 32, C = 64;
    const int M = 1088;  // Padded from 1024
    const int K = 576;   // 3x3x64
    const int N = 64;    // Output channels

    // Allocate buffers
    int8_t* input = (int8_t*)aligned_alloc(64, H * W * C);
    int8_t* im2col = (int8_t*)aligned_alloc(64, M * K);
    int8_t* filter = (int8_t*)aligned_alloc(64, K * N);
    int8_t* output = (int8_t*)aligned_alloc(64, M * N);
    int8_t* output_ref = (int8_t*)aligned_alloc(64, M * N);
    int32_t* scale = (int32_t*)aligned_alloc(64, N * sizeof(int32_t));
    int32_t* bias = (int32_t*)aligned_alloc(64, N * sizeof(int32_t));

    // Initialize with random-ish data
    for (int i = 0; i < H * W * C; i++) {
        input[i] = (i % 64) - 32;  // [-32, 31]
    }
    for (int i = 0; i < K * N; i++) {
        filter[i] = (i % 16) - 8;  // [-8, 7]
    }
    for (int i = 0; i < N; i++) {
        scale[i] = 256;  // ~1.0 in fixed point (scale >> 16)
        bias[i] = 0;
    }

    // Do im2col transform
    memset(im2col, 0, M * K);
    im2col_3x3(input, im2col, H, W, C);

    printf("Testing conv3x3_relu...\n");
    printf("  Input: %dx%dx%d -> im2col: %dx%d\n", H, W, C, M, K);
    printf("  Filter: %dx%d\n", K, N);
    printf("  Output: %dx%d\n", M, N);

    // Reference computation
    conv3x3_relu_ref(im2col, filter, scale, bias, output_ref, H * W, K, N);

    // Kernel computation
    memset(output, 0, M * N);
    conv3x3_relu(MEMREF_I8(im2col, M, K),
                 MEMREF_I8(filter, K, N),
                 MEMREF_I8(output, M, N),
                 MEMREF_I32(scale, N, 1),
                 MEMREF_I32(bias, N, 1));

    // Verify
    int errors = 0;
    for (int i = 0; i < H * W; i++) {  // Only check actual data, not padding
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            if (output[idx] != output_ref[idx]) {
                if (errors < 10) {
                    printf("  Mismatch at [%d,%d]: got %d, expected %d\n",
                           i, j, output[idx], output_ref[idx]);
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("  PASSED: Output matches reference\n");
    } else {
        printf("  FAILED: %d mismatches\n", errors);
    }

    // Benchmark
    printf("\nBenchmarking...\n");
    const int iters = 100;

    // Warmup
    conv3x3_relu(MEMREF_I8(im2col, M, K),
                 MEMREF_I8(filter, K, N),
                 MEMREF_I8(output, M, N),
                 MEMREF_I32(scale, N, 1),
                 MEMREF_I32(bias, N, 1));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        conv3x3_relu(MEMREF_I8(im2col, M, K),
                     MEMREF_I8(filter, K, N),
                     MEMREF_I8(output, M, N),
                     MEMREF_I32(scale, N, 1),
                     MEMREF_I32(bias, N, 1));
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double ops = 2.0 * M * K * N + 4.0 * M * N;  // matmul + requant/relu
    double giops = ops / ms / 1e6;

    printf("  conv3x3_relu: %.3f ms, %.0f GIOP/s\n", ms, giops);

    // Free
    free(input);
    free(im2col);
    free(filter);
    free(output);
    free(output_ref);
    free(scale);
    free(bias);

    dlclose(h);
    printf("\nDone!\n");
    return 0;
}
