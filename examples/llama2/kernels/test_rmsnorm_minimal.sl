// Minimal RMSNorm test
fn test_rmsnorm_minimal() -> f32 {
    var size = 4i;
    var eps = 0.00001;

    var input = array<f32>([4]);
    var weight = array<f32>([4]);
    var output = array<f32>([4]);

    // Simple initialization
    var i = 0i;
    while (i < size) {
        input[i] = 1.0;
        weight[i] = 1.0;
        output[i] = 0.0;
        i = i + 1i;
    }

    var result = rmsnorm(input, weight, output, size, eps);
    return result[0i];
}
