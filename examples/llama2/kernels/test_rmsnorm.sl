// Test RMSNorm operation
// RMSNorm: output = (input / sqrt(mean(input^2) + eps)) * weight

fn test_rmsnorm() -> f32 {
    var size = 4i;
    var eps = 0.00001;  // 1e-5

    // Allocate arrays
    var input = array<f32>([4]);
    var weight = array<f32>([4]);
    var output = array<f32>([4]);

    // Initialize input: [1.0, 2.0, 3.0, 4.0]
    input[0i] = 1.0;
    input[1i] = 2.0;
    input[2i] = 3.0;
    input[3i] = 4.0;

    // Initialize weight: [1.0, 1.0, 1.0, 1.0]
    weight[0i] = 1.0;
    weight[1i] = 1.0;
    weight[2i] = 1.0;
    weight[3i] = 1.0;

    // Initialize output to zero
    output[0i] = 0.0;
    output[1i] = 0.0;
    output[2i] = 0.0;
    output[3i] = 0.0;

    // Apply RMSNorm
    var result = rmsnorm(input, weight, output, size, eps);

    // Return first element for validation
    // Expected: 1.0 / sqrt((1^2 + 2^2 + 3^2 + 4^2) / 4 + 1e-5) * 1.0
    // = 1.0 / sqrt((1 + 4 + 9 + 16) / 4)
    // = 1.0 / sqrt(7.5)
    // ≈ 1.0 / 2.738 ≈ 0.365
    return result[0i];
}
