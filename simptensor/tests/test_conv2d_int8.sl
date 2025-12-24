// Actual INT8 Conv2D via im2col + VNNI matmul
// Input: NHWC, Filter: HWIO (height, width, in_channels, out_channels)

// 3x3 conv, 64 input channels, 64 output channels, 32x32 input
// im2col converts patches to columns:
//   - Each 3x3x64 patch becomes a row of 576 elements
//   - 32x32 input with padding=1 -> 32x32 output = 1024 patches
// Matrix shapes: (1024, 576) @ (576, 64) -> (1024, 64)

fn conv3x3_64ch(i8[] input, i8[] filter, i32[] output) -> i32 {
    // im2col'd input: 1024 patches x 576 (3*3*64)
    // Pad to avoid power-of-2: 1024->1088, but 576 is fine (not power of 2)
    i8<1088, 576> Im2col = tensor_from_array(input, 0i);

    // Filter reshaped: 576 (3*3*64) x 64 out_channels
    // Pad output channels: 64->128 to get better K utilization
    i8<576, 128> Filter = tensor_from_array(filter, 0i);

    // Output: 1088 x 128
    i32<1088, 128> Out = tensor_from_array(output, 0i);

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(Im2col, Filter, Out);

    return Out[0i, 0i];
}

// 3x3 conv, 128 channels, 16x16 input (deeper layer)
// 16x16 -> 256 patches, 3x3x128 = 1152 per patch
// Pad: 256->320
fn conv3x3_128ch(i8[] input, i8[] filter, i32[] output) -> i32 {
    i8<320, 1152> Im2col = tensor_from_array(input, 0i);
    i8<1152, 128> Filter = tensor_from_array(filter, 0i);
    i32<320, 128> Out = tensor_from_array(output, 0i);

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(Im2col, Filter, Out);

    return Out[0i, 0i];
}

// 3x3 conv, 256 channels, 8x8 input (even deeper)
// 8x8 -> 64 patches, 3x3x256 = 2304 per patch
// Pad: 64->128
fn conv3x3_256ch(i8[] input, i8[] filter, i32[] output) -> i32 {
    i8<128, 2304> Im2col = tensor_from_array(input, 0i);
    i8<2304, 256> Filter = tensor_from_array(filter, 0i);
    i32<128, 256> Out = tensor_from_array(output, 0i);

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(Im2col, Filter, Out);

    return Out[0i, 0i];
}

// 1x1 conv (pointwise) - common in ResNet/MobileNet
// 32x32x64 input, 128 output channels
// Just reshape: 1024 pixels x 64 channels @ 64 x 128 -> 1024 x 128
fn conv1x1_64to128(i8[] input, i8[] filter, i32[] output) -> i32 {
    i8<1088, 64> Input = tensor_from_array(input, 0i);
    i8<64, 128> Filter = tensor_from_array(filter, 0i);
    i32<1088, 128> Out = tensor_from_array(output, 0i);

    @parallel @tile(64, 64, 16) @lower("vnni.i8_matmul")
    tensor_matmul_out(Input, Filter, Out);

    return Out[0i, 0i];
}
