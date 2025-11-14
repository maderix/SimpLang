// Quantized matmul test: weights in W4/W2, activations in FP32
// Each group of 128 weights shares one scale and zero point

fn dequant_w4(i8[] qweights, f32[] scales, f32[] zeros,
              i64 idx, i64 group_size) -> f32 {
    var g = idx / group_size;
    var scale = scales[g];
    var zero = zeros[g];

    // Extract 4-bit value (2 values per byte)
    var byte_idx = idx / 2i;
    var qbyte_i8 = qweights[byte_idx];
    // Note: Need to convert i8 to i64 for bitwise ops
    var qbyte = qbyte_i8 + 0i;  // Implicit conversion

    var qval = 0i;
    if (idx % 2i == 0i) {
        qval = qbyte & 15i;  // Lower 4 bits
    } else {
        qval = (qbyte >> 4i) & 15i;  // Upper 4 bits
    }

    // Convert to float by adding 0.0
    var qval_f = qval + 0.0;
    return qval_f * scale + zero;
}

fn dequant_w2(i8[] qweights, f32[] scales, f32[] zeros,
              i64 idx, i64 group_size) -> f32 {
    var g = idx / group_size;
    var scale = scales[g];
    var zero = zeros[g];

    // Extract 2-bit value (4 values per byte)
    var byte_idx = idx / 4i;
    var qbyte_i8 = qweights[byte_idx];
    var qbyte = qbyte_i8 + 0i;  // Convert to i64

    var shift = (idx % 4i) * 2i;
    var qval = (qbyte >> shift) & 3i;

    // Convert to float
    var qval_f = qval + 0.0;
    return qval_f * scale + zero;
}

fn matmul_w4(f32[] act, i8[] qweights, f32[] scales, f32[] zeros,
             f32[] output, i64 M, i64 K, i64 N, i64 group_size) -> f32 {
    // act: [M, K] in FP32
    // qweights: [K, N] in W4 (quantized)
    // output: [M, N] in FP32

    var i = 0i;
    while (i < M) {
        var j = 0i;
        while (j < N) {
            var sum = 0.0;
            var k = 0i;
            while (k < K) {
                var w_idx = k * N + j;
                var w_val = dequant_w4(qweights, scales, zeros, w_idx, group_size);
                sum = sum + act[i * K + k] * w_val;
                k = k + 1i;
            }
            output[i * N + j] = sum;
            j = j + 1i;
        }
        i = i + 1i;
    }

    return output[0];
}

fn matmul_w2(f32[] act, i8[] qweights, f32[] scales, f32[] zeros,
             f32[] output, i64 M, i64 K, i64 N, i64 group_size) -> f32 {
    // act: [M, K] in FP32
    // qweights: [K, N] in W2 (quantized)
    // output: [M, N] in FP32

    var i = 0i;
    while (i < M) {
        var j = 0i;
        while (j < N) {
            var sum = 0.0;
            var k = 0i;
            while (k < K) {
                var w_idx = k * N + j;
                var w_val = dequant_w2(qweights, scales, zeros, w_idx, group_size);
                sum = sum + act[i * K + k] * w_val;
                k = k + 1i;
            }
            output[i * N + j] = sum;
            j = j + 1i;
        }
        i = i + 1i;
    }

    return output[0];
}
