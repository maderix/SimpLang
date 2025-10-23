// Test basic Conv2D operation with simple parameters
// Input: 1 batch, 4x4 spatial, 1 channel
// Kernel: 3x3, stride=1, padding=0
// Expected output: 1x2x2x1

fn test_conv2d_simple() -> f32 {
    // Input: 1x4x4x1 (NHWC)
    var batch = 1i;
    var in_h = 4i;
    var in_w = 4i;
    var in_c = 1i;

    // Kernel: 3x3, 1 output channel
    var out_c = 1i;
    var k_h = 3i;
    var k_w = 3i;

    // Stride and padding
    var stride_h = 1i;
    var stride_w = 1i;
    var pad_h = 0i;
    var pad_w = 0i;

    // Output dimensions: (4 + 2*0 - 3) / 1 + 1 = 2
    var out_h = 2i;
    var out_w = 2i;

    // Allocate arrays
    var input_size = 16i;  // 1*4*4*1
    var weight_size = 9i;  // 1*3*3*1
    var bias_size = 1i;
    var output_size = 4i;  // 1*2*2*1

    var input = array<f32>([16]);
    var weights = array<f32>([9]);
    var bias = array<f32>([1]);
    var output = array<f32>([4]);

    // Initialize input with simple values (row-major NHWC)
    var i = 0i;
    while (i < input_size) {
        input[i] = i;
        i = i + 1i;
    }

    // Initialize weights with ones
    i = 0i;
    while (i < weight_size) {
        weights[i] = 1.0;
        i = i + 1i;
    }

    // Initialize bias to zero
    bias[0i] = 0.0;

    // Initialize output to zero
    i = 0i;
    while (i < output_size) {
        output[i] = 0.0;
        i = i + 1i;
    }

    // Perform Conv2D
    var result = conv2d(input, weights, bias, output,
                        batch, in_h, in_w, in_c,
                        out_c, k_h, k_w,
                        stride_h, stride_w, pad_h, pad_w);

    // Return first output element for validation
    return result[0i];
}
