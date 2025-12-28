// Test SiLU (Swish) activation: x / (1 + exp(-x))
fn test_silu() -> f32 {
    var size = 4i;

    var input = array<f32>([4]);
    var output = array<f32>([4]);

    // Initialize input with values
    input[0i] = 1.0;
    input[1i] = 2.0;
    input[2i] = -1.0;
    input[3i] = 0.0;

    output[0i] = 0.0;
    output[1i] = 0.0;
    output[2i] = 0.0;
    output[3i] = 0.0;

    var result = silu(input, output, size);
    return result[0i];
}
