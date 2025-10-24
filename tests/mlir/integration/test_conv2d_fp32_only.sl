// Simple FP32 Conv2D test
fn bench_conv2d_fp32_small(f32[] input, f32[] weights, f32[] bias, f32[] output, i64 batch, i64 in_h, i64 in_w, i64 in_c, i64 out_c, i64 k_h, i64 k_w) -> f32 {
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
