#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <chrono>

// 2D memref: base, aligned, offset, size0 (rows), size1 (cols)
#define MEMREF_I8(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)
#define MEMREF_I32(ptr, m, n) (ptr), (ptr), (int64_t)0, (int64_t)(m), (int64_t)(n)

typedef int32_t (*ConvFn)(int8_t*, int8_t*, int64_t, int64_t, int64_t,
                          int8_t*, int8_t*, int64_t, int64_t, int64_t,
                          int32_t*, int32_t*, int64_t, int64_t, int64_t);

// Actual im2col implementation for 3x3 conv with padding=1
void im2col_3x3(const int8_t* input, int8_t* col,
                int H, int W, int C, int pad_patches) {
    int out_H = H;  // same padding
    int out_W = W;
    int patch_size = 9 * C;  // 3x3*C

    memset(col, 0, pad_patches * patch_size);

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
                        }
                        col_offset++;
                    }
                }
            }
        }
    }
}

struct ConvConfig {
    const char* name;
    const char* symbol;
    int H, W, C_in, C_out;
    int im2col_rows, im2col_cols;
    int filter_rows, filter_cols;
    int out_rows, out_cols;
    int iters;
    int kernel_size;  // 3 for 3x3, 1 for 1x1
};

double bench_conv(ConvFn fn, const ConvConfig& cfg) {
    size_t input_sz = cfg.H * cfg.W * cfg.C_in;
    int k2 = cfg.kernel_size * cfg.kernel_size;
    size_t filter_sz = k2 * cfg.C_in * cfg.C_out;
    size_t im2col_sz = (size_t)cfg.im2col_rows * cfg.im2col_cols;
    size_t filter_mat_sz = (size_t)cfg.filter_rows * cfg.filter_cols;
    size_t out_sz = (size_t)cfg.out_rows * cfg.out_cols;

    int8_t* input = (int8_t*)aligned_alloc(64, input_sz);
    int8_t* filter = (int8_t*)aligned_alloc(64, filter_mat_sz);
    int8_t* im2col = (int8_t*)aligned_alloc(64, im2col_sz);
    int32_t* output = (int32_t*)aligned_alloc(64, out_sz * 4);

    // Initialize with random-ish data
    for (size_t i = 0; i < input_sz; i++) input[i] = (i % 127) - 63;
    for (size_t i = 0; i < filter_mat_sz; i++) filter[i] = (i % 63) - 31;
    memset(output, 0, out_sz * 4);

    // Do im2col transform (only for 3x3 convs)
    if (cfg.kernel_size == 3) {
        im2col_3x3(input, im2col, cfg.H, cfg.W, cfg.C_in, cfg.im2col_rows);
    } else {
        // 1x1 conv: just reshape input to (H*W, C_in)
        memcpy(im2col, input, im2col_sz);
    }

    // Warmup - pass 2D dimensions (rows, cols) for each memref
    fn(MEMREF_I8(im2col, cfg.im2col_rows, cfg.im2col_cols),
       MEMREF_I8(filter, cfg.filter_rows, cfg.filter_cols),
       MEMREF_I32(output, cfg.out_rows, cfg.out_cols));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.iters; i++) {
        fn(MEMREF_I8(im2col, cfg.im2col_rows, cfg.im2col_cols),
           MEMREF_I8(filter, cfg.filter_rows, cfg.filter_cols),
           MEMREF_I32(output, cfg.out_rows, cfg.out_cols));
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / cfg.iters;

    // GIOP/s = 2*M*K*N / time_ms / 1e6
    double giops = 2.0 * cfg.im2col_rows * cfg.im2col_cols * cfg.filter_cols / ms / 1e6;

    // Also compute effective conv GFLOPS (actual convolution ops)
    // Conv ops = 2 * H_out * W_out * K_h * K_w * C_in * C_out
    double conv_ops = 2.0 * cfg.H * cfg.W * k2 * cfg.C_in * cfg.C_out;
    double conv_gflops = conv_ops / ms / 1e6;

    free(input);
    free(filter);
    free(im2col);
    free(output);

    return giops;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <conv2d_int8.so>\n", argv[0]);
        return 1;
    }
    void* h = dlopen(argv[1], RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }

    printf("=== INT8 Conv2D Benchmark (im2col + VNNI matmul) ===\n\n");
    printf("%-20s %-25s %-20s %10s\n", "Config", "Conv Shape", "Matmul Shape", "GIOP/s");
    printf("--------------------------------------------------------------------------------\n");

    // Conv configs: name, symbol, H, W, C_in, C_out, im2col dims, filter dims, out dims, iters, kernel_size
    ConvConfig configs[] = {
        // 3x3 conv, 64ch, 32x32 input -> 1024 patches, 576 per patch
        {"3x3 64ch 32x32", "conv3x3_64ch", 32, 32, 64, 64,
         1088, 576, 576, 128, 1088, 128, 100, 3},

        // 3x3 conv, 128ch, 16x16 input -> 256 patches, 1152 per patch
        {"3x3 128ch 16x16", "conv3x3_128ch", 16, 16, 128, 128,
         320, 1152, 1152, 128, 320, 128, 200, 3},

        // 3x3 conv, 256ch, 8x8 input -> 64 patches, 2304 per patch
        {"3x3 256ch 8x8", "conv3x3_256ch", 8, 8, 256, 256,
         128, 2304, 2304, 256, 128, 256, 500, 3},

        // 1x1 conv (pointwise), 64->128ch, 32x32 input
        {"1x1 64->128 32x32", "conv1x1_64to128", 32, 32, 64, 128,
         1088, 64, 64, 128, 1088, 128, 200, 1},
    };

    for (const auto& cfg : configs) {
        ConvFn fn = (ConvFn)dlsym(h, cfg.symbol);
        if (!fn) {
            printf("%-20s symbol not found\n", cfg.name);
            continue;
        }

        double giops = bench_conv(fn, cfg);

        char conv_shape[64], mat_shape[64];
        snprintf(conv_shape, sizeof(conv_shape), "%dx%dx%d->%d (%dx%d)",
                 cfg.H, cfg.W, cfg.C_in, cfg.C_out, cfg.kernel_size, cfg.kernel_size);
        snprintf(mat_shape, sizeof(mat_shape), "(%d,%d)@(%d,%d)",
                 cfg.im2col_rows, cfg.im2col_cols, cfg.filter_rows, cfg.filter_cols);

        printf("%-20s %-25s %-20s %10.0f\n", cfg.name, conv_shape, mat_shape, giops);
    }

    printf("\nNote: Using padded dimensions to avoid cache aliasing\n");
    printf("      im2col transform included in setup (not timed)\n");

    dlclose(h);
    return 0;
}
