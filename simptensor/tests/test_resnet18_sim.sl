// ResNet18 INT8 Simulation
// All convs use im2col + VNNI matmul
// Input: 224x224x3 -> Output: 1000 classes

// Layer dimensions (after im2col):
// conv1: 7x7 stride 2 -> 112x112 output, 3->64 channels
// layer1: 56x56, 64ch (after maxpool)
// layer2: 28x28, 128ch
// layer3: 14x14, 256ch
// layer4: 7x7, 512ch

// === Layer 1 Block (56x56, 64ch) ===
fn layer1_block(i8[] im2col, i8[] filter1, i8[] filter2,
                i32[] scale1, i32[] bias1,
                i32[] scale2, i32[] bias2,
                i8[] residual, i8[] output) -> i32 {
    // im2col: 56*56=3136 patches, pad to 3200, K=576 (3x3x64)
    i8<3200, 576> A1 = tensor_from_array(im2col, 0i);
    i8<576, 64> B1 = tensor_from_array(filter1, 0i);
    i32<3200, 64> C1;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A1, B1, C1);

    // Requant + ReLU for conv1
    i32<64> Scale1 = tensor_from_array(scale1, 0i);
    i32<64> Bias1 = tensor_from_array(bias1, 0i);
    i8<3200, 64> Inter;

    for (var i = 0i; i < 3200i; i = i + 1i) {
        for (var j = 0i; j < 64i; j = j + 1i) {
            var acc = C1[i, j] as i64;
            var scaled = (acc * (Scale1[j] as i64)) / 65536i;
            var biased = scaled + (Bias1[j] as i64);
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            Inter[i, j] = biased as i8;
        }
    }

    // Conv2 (would need im2col of Inter, simplified here)
    i8<576, 64> B2 = tensor_from_array(filter2, 0i);
    i32<3200, 64> C2;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A1, B2, C2);  // Reusing A1 for demo

    // Requant + residual add + ReLU
    i32<64> Scale2 = tensor_from_array(scale2, 0i);
    i32<64> Bias2 = tensor_from_array(bias2, 0i);
    i8<3200, 64> Res = tensor_from_array(residual, 0i);
    i8<3200, 64> Out = tensor_from_array(output, 0i);

    for (var i = 0i; i < 3200i; i = i + 1i) {
        for (var j = 0i; j < 64i; j = j + 1i) {
            var acc = C2[i, j] as i64;
            var scaled = (acc * (Scale2[j] as i64)) / 65536i;
            var conv_out = scaled + (Bias2[j] as i64);
            var res = Res[i, j] as i64;
            var sum = conv_out + res;
            if (sum < 0i) { sum = 0i; }
            if (sum > 127i) { sum = 127i; }
            Out[i, j] = sum as i8;
        }
    }

    return 0i;
}

// === Layer 2 Block (28x28, 128ch) ===
fn layer2_block(i8[] im2col, i8[] filter1, i8[] filter2,
                i32[] scale1, i32[] bias1,
                i32[] scale2, i32[] bias2,
                i8[] residual, i8[] output) -> i32 {
    // 28*28=784 patches, pad to 832, K=1152 (3x3x128)
    i8<832, 1152> A1 = tensor_from_array(im2col, 0i);
    i8<1152, 128> B1 = tensor_from_array(filter1, 0i);
    i32<832, 128> C1;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A1, B1, C1);

    i32<128> Scale1 = tensor_from_array(scale1, 0i);
    i32<128> Bias1 = tensor_from_array(bias1, 0i);
    i8<832, 128> Inter;

    for (var i = 0i; i < 832i; i = i + 1i) {
        for (var j = 0i; j < 128i; j = j + 1i) {
            var acc = C1[i, j] as i64;
            var scaled = (acc * (Scale1[j] as i64)) / 65536i;
            var biased = scaled + (Bias1[j] as i64);
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            Inter[i, j] = biased as i8;
        }
    }

    i8<1152, 128> B2 = tensor_from_array(filter2, 0i);
    i32<832, 128> C2;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A1, B2, C2);

    i32<128> Scale2 = tensor_from_array(scale2, 0i);
    i32<128> Bias2 = tensor_from_array(bias2, 0i);
    i8<832, 128> Res = tensor_from_array(residual, 0i);
    i8<832, 128> Out = tensor_from_array(output, 0i);

    for (var i = 0i; i < 832i; i = i + 1i) {
        for (var j = 0i; j < 128i; j = j + 1i) {
            var acc = C2[i, j] as i64;
            var scaled = (acc * (Scale2[j] as i64)) / 65536i;
            var conv_out = scaled + (Bias2[j] as i64);
            var res = Res[i, j] as i64;
            var sum = conv_out + res;
            if (sum < 0i) { sum = 0i; }
            if (sum > 127i) { sum = 127i; }
            Out[i, j] = sum as i8;
        }
    }

    return 0i;
}

// === Layer 3 Block (14x14, 256ch) ===
fn layer3_block(i8[] im2col, i8[] filter1, i8[] filter2,
                i32[] scale1, i32[] bias1,
                i32[] scale2, i32[] bias2,
                i8[] residual, i8[] output) -> i32 {
    // 14*14=196 patches, pad to 256, K=2304 (3x3x256)
    i8<256, 2304> A1 = tensor_from_array(im2col, 0i);
    i8<2304, 256> B1 = tensor_from_array(filter1, 0i);
    i32<256, 256> C1;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A1, B1, C1);

    i32<256> Scale1 = tensor_from_array(scale1, 0i);
    i32<256> Bias1 = tensor_from_array(bias1, 0i);
    i8<256, 256> Inter;

    for (var i = 0i; i < 256i; i = i + 1i) {
        for (var j = 0i; j < 256i; j = j + 1i) {
            var acc = C1[i, j] as i64;
            var scaled = (acc * (Scale1[j] as i64)) / 65536i;
            var biased = scaled + (Bias1[j] as i64);
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            Inter[i, j] = biased as i8;
        }
    }

    i8<2304, 256> B2 = tensor_from_array(filter2, 0i);
    i32<256, 256> C2;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A1, B2, C2);

    i32<256> Scale2 = tensor_from_array(scale2, 0i);
    i32<256> Bias2 = tensor_from_array(bias2, 0i);
    i8<256, 256> Res = tensor_from_array(residual, 0i);
    i8<256, 256> Out = tensor_from_array(output, 0i);

    for (var i = 0i; i < 256i; i = i + 1i) {
        for (var j = 0i; j < 256i; j = j + 1i) {
            var acc = C2[i, j] as i64;
            var scaled = (acc * (Scale2[j] as i64)) / 65536i;
            var conv_out = scaled + (Bias2[j] as i64);
            var res = Res[i, j] as i64;
            var sum = conv_out + res;
            if (sum < 0i) { sum = 0i; }
            if (sum > 127i) { sum = 127i; }
            Out[i, j] = sum as i8;
        }
    }

    return 0i;
}

// === Layer 4 Block (7x7, 512ch) ===
fn layer4_block(i8[] im2col, i8[] filter1, i8[] filter2,
                i32[] scale1, i32[] bias1,
                i32[] scale2, i32[] bias2,
                i8[] residual, i8[] output) -> i32 {
    // 7*7=49 patches, pad to 64, K=4608 (3x3x512)
    i8<64, 4608> A1 = tensor_from_array(im2col, 0i);
    i8<4608, 512> B1 = tensor_from_array(filter1, 0i);
    i32<64, 512> C1;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A1, B1, C1);

    i32<512> Scale1 = tensor_from_array(scale1, 0i);
    i32<512> Bias1 = tensor_from_array(bias1, 0i);
    i8<64, 512> Inter;

    for (var i = 0i; i < 64i; i = i + 1i) {
        for (var j = 0i; j < 512i; j = j + 1i) {
            var acc = C1[i, j] as i64;
            var scaled = (acc * (Scale1[j] as i64)) / 65536i;
            var biased = scaled + (Bias1[j] as i64);
            if (biased < 0i) { biased = 0i; }
            if (biased > 127i) { biased = 127i; }
            Inter[i, j] = biased as i8;
        }
    }

    i8<4608, 512> B2 = tensor_from_array(filter2, 0i);
    i32<64, 512> C2;

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(A1, B2, C2);

    i32<512> Scale2 = tensor_from_array(scale2, 0i);
    i32<512> Bias2 = tensor_from_array(bias2, 0i);
    i8<64, 512> Res = tensor_from_array(residual, 0i);
    i8<64, 512> Out = tensor_from_array(output, 0i);

    for (var i = 0i; i < 64i; i = i + 1i) {
        for (var j = 0i; j < 512i; j = j + 1i) {
            var acc = C2[i, j] as i64;
            var scaled = (acc * (Scale2[j] as i64)) / 65536i;
            var conv_out = scaled + (Bias2[j] as i64);
            var res = Res[i, j] as i64;
            var sum = conv_out + res;
            if (sum < 0i) { sum = 0i; }
            if (sum > 127i) { sum = 127i; }
            Out[i, j] = sum as i8;
        }
    }

    return 0i;
}

// === Final FC layer (512 -> 1000) ===
fn fc_layer(i8[] input, i8[] weight, i32[] scale, i32[] bias, i32[] output) -> i32 {
    // After global avgpool: 1x512, weight: 512x1000
    // Pad to 1x512 @ 512x1024
    i8<1, 512> A = tensor_from_array(input, 0i);
    i8<512, 1024> B = tensor_from_array(weight, 0i);
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
