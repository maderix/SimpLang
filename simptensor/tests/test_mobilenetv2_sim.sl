// MobileNetV2 INT8 Simulation
// Uses depthwise separable convolutions with inverted residuals
// Input: 224x224x3 -> Output: 1000 classes

// Inverted Residual Block:
// 1. Expand: 1x1 conv (channels -> channels*expansion)
// 2. Depthwise: 3x3 depthwise conv (each channel separate)
// 3. Project: 1x1 conv (channels*expansion -> channels)

// === Pointwise Conv 1x1 (used for expand/project) ===
fn pointwise_conv(i8[] input, i8[] filter, i8[] output,
                  i32[] scale, i32[] bias,
                  i64 M, i64 C_in, i64 C_out) -> i32 {
    // 1x1 conv is just matmul: (M, C_in) @ (C_in, C_out)
    // Using fixed sizes for layer1 expand: 56x56=3136->3200, 32->192
    i8<3200, 32> A = tensor_from_array(input, 0i);
    i8<32, 192> B = tensor_from_array(filter, 0i);
    i32<3200, 192> C;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A, B, C);

    i32<192> Scale = tensor_from_array(scale, 0i);
    i32<192> Bias = tensor_from_array(bias, 0i);
    i8<3200, 192> Out = tensor_from_array(output, 0i);

    for (var i = 0i; i < 3200i; i = i + 1i) {
        for (var j = 0i; j < 192i; j = j + 1i) {
            var acc = C[i, j] as i64;
            var scaled = (acc * (Scale[j] as i64)) / 65536i;
            var biased = scaled + (Bias[j] as i64);
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            Out[i, j] = biased as i8;
        }
    }

    return 0i;
}

// === Depthwise Conv 3x3 (channel-wise) ===
// For depthwise, each channel is processed independently
// im2col gives (M, 9) per channel, filter is (9, 1) per channel
fn depthwise_conv(i8[] input, i8[] filter, i8[] output,
                  i32[] scale, i32[] bias) -> i32 {
    // 56x56 spatial, 192 channels after expansion
    // Depthwise: each of 192 channels has 3x3=9 weights
    // Treat as (M*C, 9) @ (9, 1) but that's inefficient
    // Better: loop over channels, do element-wise 3x3 conv

    // Simplified: treat as (3200, 9*192) @ (9*192, 192) with sparse structure
    // But actually for demo, we'll use a grouped approach
    // Here we just simulate the compute as regular conv for timing
    i8<3200, 1728> A = tensor_from_array(input, 0i);  // 9*192=1728
    i8<1728, 192> B = tensor_from_array(filter, 0i);
    i32<3200, 192> C;

    // Note: Real depthwise would be sparse, this is dense approximation
    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A, B, C);

    i32<192> Scale = tensor_from_array(scale, 0i);
    i32<192> Bias = tensor_from_array(bias, 0i);
    i8<3200, 192> Out = tensor_from_array(output, 0i);

    for (var i = 0i; i < 3200i; i = i + 1i) {
        for (var j = 0i; j < 192i; j = j + 1i) {
            var acc = C[i, j] as i64;
            var scaled = (acc * (Scale[j] as i64)) / 65536i;
            var biased = scaled + (Bias[j] as i64);
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            Out[i, j] = biased as i8;
        }
    }

    return 0i;
}

// === Inverted Residual Block (expansion=6) ===
// Layer 1: 56x56, 32ch -> expand to 192 -> depthwise -> project to 32
// TODO: K=32 falls back to scalar (VNNI requires K>=64). Fix VNNIPass to handle K<64 efficiently.
fn inverted_residual_32ch(i8[] input,
                          i8[] expand_filter, i32[] expand_scale, i32[] expand_bias,
                          i8[] dw_filter, i32[] dw_scale, i32[] dw_bias,
                          i8[] project_filter, i32[] project_scale, i32[] project_bias,
                          i8[] output) -> i32 {
    // Expand: 1x1 conv 32->192
    i8<3200, 32> In = tensor_from_array(input, 0i);
    i8<32, 192> ExpF = tensor_from_array(expand_filter, 0i);
    i32<3200, 192> Exp_i32;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(In, ExpF, Exp_i32);

    i32<192> ExpScale = tensor_from_array(expand_scale, 0i);
    i32<192> ExpBias = tensor_from_array(expand_bias, 0i);
    i8<3200, 192> Expanded;

    for (var i = 0i; i < 3200i; i = i + 1i) {
        for (var j = 0i; j < 192i; j = j + 1i) {
            var acc = Exp_i32[i, j] as i64;
            var scaled = (acc * (ExpScale[j] as i64)) / 65536i;
            var biased = scaled + (ExpBias[j] as i64);
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            Expanded[i, j] = biased as i8;
        }
    }

    // Depthwise 3x3 as padded dense matmul
    // Real depthwise: (M, 9*C) @ block_diag(9x1 per channel)
    // Approximation: (M, C) @ (C, C) dense matmul with ~9x compute
    i8<192, 192> DwF = tensor_from_array(dw_filter, 0i);
    i32<3200, 192> Dw_i32;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(Expanded, DwF, Dw_i32);

    i32<192> DwScale = tensor_from_array(dw_scale, 0i);
    i32<192> DwBias = tensor_from_array(dw_bias, 0i);
    i8<3200, 192> DwOut;

    for (var i = 0i; i < 3200i; i = i + 1i) {
        for (var j = 0i; j < 192i; j = j + 1i) {
            var acc = Dw_i32[i, j] as i64;
            var scaled = (acc * (DwScale[j] as i64)) / 65536i;
            var biased = scaled + (DwBias[j] as i64);
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            DwOut[i, j] = biased as i8;
        }
    }

    // Project: 1x1 conv 192->32
    i8<192, 32> ProjF = tensor_from_array(project_filter, 0i);
    i32<3200, 32> Proj_i32;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(DwOut, ProjF, Proj_i32);

    i32<32> ProjScale = tensor_from_array(project_scale, 0i);
    i32<32> ProjBias = tensor_from_array(project_bias, 0i);
    i8<3200, 32> Residual = tensor_from_array(input, 0i);
    i8<3200, 32> Out = tensor_from_array(output, 0i);

    // Project + residual (no ReLU on project in MobileNetV2)
    for (var i = 0i; i < 3200i; i = i + 1i) {
        for (var j = 0i; j < 32i; j = j + 1i) {
            var acc = Proj_i32[i, j] as i64;
            var scaled = (acc * (ProjScale[j] as i64)) / 65536i;
            var proj = scaled + (ProjBias[j] as i64);
            var res = Residual[i, j] as i64;
            var sum = proj + res;
            // No ReLU, just clamp
            if (sum < -128i) { sum = -128i; }
            if (sum > 127i) { sum = 127i; }
            Out[i, j] = sum as i8;
        }
    }

    return 0i;
}

// === Inverted Residual 64ch (28x28 spatial) ===
fn inverted_residual_64ch(i8[] input,
                          i8[] expand_filter, i32[] expand_scale, i32[] expand_bias,
                          i8[] dw_filter, i32[] dw_scale, i32[] dw_bias,
                          i8[] project_filter, i32[] project_scale, i32[] project_bias,
                          i8[] output) -> i32 {
    // 28x28=784 -> pad to 832, expand 64->384
    i8<832, 64> In = tensor_from_array(input, 0i);
    i8<64, 384> ExpF = tensor_from_array(expand_filter, 0i);
    i32<832, 384> Exp_i32;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(In, ExpF, Exp_i32);

    i32<384> ExpScale = tensor_from_array(expand_scale, 0i);
    i32<384> ExpBias = tensor_from_array(expand_bias, 0i);
    i8<832, 384> Expanded;

    for (var i = 0i; i < 832i; i = i + 1i) {
        for (var j = 0i; j < 384i; j = j + 1i) {
            var acc = Exp_i32[i, j] as i64;
            var scaled = (acc * (ExpScale[j] as i64)) / 65536i;
            var biased = scaled + (ExpBias[j] as i64);
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            Expanded[i, j] = biased as i8;
        }
    }

    // Project: 384->64
    i8<384, 64> ProjF = tensor_from_array(project_filter, 0i);
    i32<832, 64> Proj_i32;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(Expanded, ProjF, Proj_i32);

    i32<64> ProjScale = tensor_from_array(project_scale, 0i);
    i32<64> ProjBias = tensor_from_array(project_bias, 0i);
    i8<832, 64> Residual = tensor_from_array(input, 0i);
    i8<832, 64> Out = tensor_from_array(output, 0i);

    for (var i = 0i; i < 832i; i = i + 1i) {
        for (var j = 0i; j < 64i; j = j + 1i) {
            var acc = Proj_i32[i, j] as i64;
            var scaled = (acc * (ProjScale[j] as i64)) / 65536i;
            var proj = scaled + (ProjBias[j] as i64);
            var res = Residual[i, j] as i64;
            var sum = proj + res;
            if (sum < -128i) { sum = -128i; }
            if (sum > 127i) { sum = 127i; }
            Out[i, j] = sum as i8;
        }
    }

    return 0i;
}

// === Inverted Residual 96ch (14x14 spatial) ===
fn inverted_residual_96ch(i8[] input,
                          i8[] expand_filter, i32[] expand_scale, i32[] expand_bias,
                          i8[] project_filter, i32[] project_scale, i32[] project_bias,
                          i8[] output) -> i32 {
    // 14x14=196 -> pad to 256, expand 96->576
    i8<256, 96> In = tensor_from_array(input, 0i);
    i8<96, 576> ExpF = tensor_from_array(expand_filter, 0i);
    i32<256, 576> Exp_i32;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(In, ExpF, Exp_i32);

    i32<576> ExpScale = tensor_from_array(expand_scale, 0i);
    i32<576> ExpBias = tensor_from_array(expand_bias, 0i);
    i8<256, 576> Expanded;

    for (var i = 0i; i < 256i; i = i + 1i) {
        for (var j = 0i; j < 576i; j = j + 1i) {
            var acc = Exp_i32[i, j] as i64;
            var scaled = (acc * (ExpScale[j] as i64)) / 65536i;
            var biased = scaled + (ExpBias[j] as i64);
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            Expanded[i, j] = biased as i8;
        }
    }

    // Project: 576->96
    i8<576, 96> ProjF = tensor_from_array(project_filter, 0i);
    i32<256, 96> Proj_i32;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(Expanded, ProjF, Proj_i32);

    i32<96> ProjScale = tensor_from_array(project_scale, 0i);
    i32<96> ProjBias = tensor_from_array(project_bias, 0i);
    i8<256, 96> Residual = tensor_from_array(input, 0i);
    i8<256, 96> Out = tensor_from_array(output, 0i);

    for (var i = 0i; i < 256i; i = i + 1i) {
        for (var j = 0i; j < 96i; j = j + 1i) {
            var acc = Proj_i32[i, j] as i64;
            var scaled = (acc * (ProjScale[j] as i64)) / 65536i;
            var proj = scaled + (ProjBias[j] as i64);
            var res = Residual[i, j] as i64;
            var sum = proj + res;
            if (sum < -128i) { sum = -128i; }
            if (sum > 127i) { sum = 127i; }
            Out[i, j] = sum as i8;
        }
    }

    return 0i;
}

// === Final classifier (1280 -> 1000) ===
fn classifier(i8[] input, i8[] weight, i32[] scale, i32[] bias, i32[] output) -> i32 {
    i8<1, 1280> A = tensor_from_array(input, 0i);
    i8<1280, 1024> B = tensor_from_array(weight, 0i);
    i32<1, 1024> C;

    @parallel @tile(1, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A, B, C);

    i32<1024> Scale = tensor_from_array(scale, 0i);
    i32<1024> Bias = tensor_from_array(bias, 0i);
    i32<1, 1024> Out = tensor_from_array(output, 0i);

    for (var j = 0i; j < 1024i; j = j + 1i) {
        var acc = C[0i, j] as i64;
        var scaled = (acc * (Scale[j] as i64)) / 65536i;
        Out[0i, j] = (scaled + (Bias[j] as i64)) as i32;
    }

    return 0i;
}
