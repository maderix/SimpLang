// Conv2D Benchmark for 4K Image Processing
// Production-realistic: Host prepares data, kernel does compute

// ========== FP32 CONVOLUTION ==========

// 512x512 RGB -> 512x512x64 (3x3 kernel, SAME padding)
fn bench_conv2d_fp32_small(f32[] input, f32[] weights, f32[] bias, f32[] output, i64 batch, i64 in_h, i64 in_w, i64 in_c, i64 out_c, i64 k_h, i64 k_w) -> f32 {
    var stride_h = 1i;
    var stride_w = 1i;
    var pad_h = 1i;  // SAME padding
    var pad_w = 1i;

    var result = conv2d(input, weights, bias, output,
                        batch, in_h, in_w, in_c,
                        out_c, k_h, k_w,
                        stride_h, stride_w, pad_h, pad_w);

    return result[0i];
}

// 1024x1024 RGB -> 1024x1024x32 (3x3 kernel, SAME padding)
fn bench_conv2d_fp32_medium(f32[] input, f32[] weights, f32[] bias, f32[] output, i64 batch, i64 in_h, i64 in_w, i64 in_c, i64 out_c, i64 k_h, i64 k_w) -> f32 {
    var stride_h = 1i;
    var stride_w = 1i;
    var pad_h = 1i;
    var pad_w = 1i;

    var result = conv2d(input, weights, bias, output,
                        batch, in_h, in_w, in_c,
                        out_c, k_h, k_w,
                        stride_h, stride_w, pad_h, pad_w);

    return result[0i];
}

// ========== FP16 CONVOLUTION ==========

fn bench_conv2d_fp16_small(f16[] input, f16[] weights, f16[] bias, f16[] output, i64 batch, i64 in_h, i64 in_w, i64 in_c, i64 out_c, i64 k_h, i64 k_w) -> f16 {
    var stride_h = 1i;
    var stride_w = 1i;
    var pad_h = 1i;
    var pad_w = 1i;

    var result = conv2d(input, weights, bias, output,
                        batch, in_h, in_w, in_c,
                        out_c, k_h, k_w,
                        stride_h, stride_w, pad_h, pad_w);

    return result[0i];
}

fn bench_conv2d_fp16_medium(f16[] input, f16[] weights, f16[] bias, f16[] output, i64 batch, i64 in_h, i64 in_w, i64 in_c, i64 out_c, i64 k_h, i64 k_w) -> f16 {
    var stride_h = 1i;
    var stride_w = 1i;
    var pad_h = 1i;
    var pad_w = 1i;

    var result = conv2d(input, weights, bias, output,
                        batch, in_h, in_w, in_c,
                        out_c, k_h, k_w,
                        stride_h, stride_w, pad_h, pad_w);

    return result[0i];
}

// ========== I8 QUANTIZED CONVOLUTION ==========

// Quantized INT8 inference (simplified - same type bias)
fn bench_conv2d_i8_small_quantized(i8[] input, i8[] weights, i8[] bias, i8[] output, i64 batch, i64 in_h, i64 in_w, i64 in_c, i64 out_c, i64 k_h, i64 k_w, i32 input_zero_point, i32 output_zero_point) -> i8 {
    var stride_h = 1i;
    var stride_w = 1i;
    var pad_h = 1i;
    var pad_w = 1i;

    // Note: In real PCQ, we'd apply per-channel dequant/requant here
    // For benchmark, we do the raw integer convolution
    // Host can apply scaling before/after

    var result = conv2d(input, weights, bias, output,
                        batch, in_h, in_w, in_c,
                        out_c, k_h, k_w,
                        stride_h, stride_w, pad_h, pad_w);

    return result[0i];
}

fn bench_conv2d_i8_medium_quantized(i8[] input, i8[] weights, i8[] bias, i8[] output, i64 batch, i64 in_h, i64 in_w, i64 in_c, i64 out_c, i64 k_h, i64 k_w, i32 input_zero_point, i32 output_zero_point) -> i8 {
    var stride_h = 1i;
    var stride_w = 1i;
    var pad_h = 1i;
    var pad_w = 1i;

    var result = conv2d(input, weights, bias, output,
                        batch, in_h, in_w, in_c,
                        out_c, k_h, k_w,
                        stride_h, stride_w, pad_h, pad_w);

    return result[0i];
}
