// ResNet Basic Block - INT8 with fused Conv+ReLU
// Input: 32x32x64, Output: 32x32x64 (same spatial, same channels)
// Uses im2col + VNNI matmul approach

// Fused Conv 3x3 + ReLU: outputs clamped to [0, 127] for INT8
fn conv3x3_relu(i8[] im2col, i8[] filter, i8[] output,
                i32[] scale, i32[] bias) -> i32 {
    // im2col input: 1024 patches x 576 (3x3x64)
    // Padded to 1088 to avoid cache aliasing
    i8<1088, 576> A = tensor_from_array(im2col, 0i);
    i8<576, 64> B = tensor_from_array(filter, 0i);
    i32<1088, 64> C_i32;  // Accumulator

    // INT8 matmul -> i32 accumulator
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A, B, C_i32);

    // Requantize + ReLU: scale, add bias, clamp to [0, 127]
    i32<64> Scale = tensor_from_array(scale, 0i);
    i32<64> Bias = tensor_from_array(bias, 0i);
    i8<1088, 64> Out = tensor_from_array(output, 0i);

    // Fused requant + ReLU
    for (var i = 0i; i < 1088i; i = i + 1i) {
        for (var j = 0i; j < 64i; j = j + 1i) {
            var acc = C_i32[i, j] as i64;
            var scaled = (acc * (Scale[j] as i64)) / 65536i;
            var biased = scaled + (Bias[j] as i64);
            // ReLU + clamp to i8 range
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            Out[i, j] = biased as i8;
        }
    }

    return 0i;
}

// Conv 3x3 without activation (for second conv before residual add)
fn conv3x3_linear(i8[] im2col, i8[] filter, i32[] output,
                  i32[] scale, i32[] bias) -> i32 {
    i8<1088, 576> A = tensor_from_array(im2col, 0i);
    i8<576, 64> B = tensor_from_array(filter, 0i);
    i32<1088, 64> C_i32;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A, B, C_i32);

    // Requantize only (no ReLU) - output stays i32 for residual add
    i32<64> Scale = tensor_from_array(scale, 0i);
    i32<64> Bias = tensor_from_array(bias, 0i);
    i32<1088, 64> Out = tensor_from_array(output, 0i);

    for (var i = 0i; i < 1088i; i = i + 1i) {
        for (var j = 0i; j < 64i; j = j + 1i) {
            var acc = C_i32[i, j];
            var scaled = (acc * Scale[j]) / 65536i;
            Out[i, j] = scaled + Bias[j];
        }
    }

    return 0i;
}

// Residual add + ReLU: output = ReLU(conv_out + residual)
fn residual_add_relu(i32[] conv_out, i8[] residual, i8[] output) -> i32 {
    i32<1088, 64> ConvOut = tensor_from_array(conv_out, 0i);
    i8<1088, 64> Residual = tensor_from_array(residual, 0i);
    i8<1088, 64> Out = tensor_from_array(output, 0i);

    for (var i = 0i; i < 1088i; i = i + 1i) {
        for (var j = 0i; j < 64i; j = j + 1i) {
            var sum = ConvOut[i, j] + (Residual[i, j] as i32);
            // ReLU + clamp - use i64 for comparisons then cast back
            var sum64 = sum as i64;
            if (sum64 < 0i) { sum64 = 0i; }
            if (sum64 > 127i) { sum64 = 127i; }
            Out[i, j] = sum64 as i8;
        }
    }

    return 0i;
}

// BatchNorm (standalone) - for inference: y = scale * x + bias
fn batchnorm_i8(i8[] input, i8[] output, i32[] scale, i32[] bias) -> i32 {
    i8<1088, 64> In = tensor_from_array(input, 0i);
    i8<1088, 64> Out = tensor_from_array(output, 0i);
    i32<64> Scale = tensor_from_array(scale, 0i);
    i32<64> Bias = tensor_from_array(bias, 0i);

    for (var i = 0i; i < 1088i; i = i + 1i) {
        for (var j = 0i; j < 64i; j = j + 1i) {
            var x = In[i, j] as i64;
            var scaled = (x * (Scale[j] as i64)) / 65536i;
            var result = scaled + (Bias[j] as i64);
            if (result < -128i) { result = -128i; }
            if (result > 127i) { result = 127i; }
            Out[i, j] = result as i8;
        }
    }

    return 0i;
}
