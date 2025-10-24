// Rotary Position Embedding (RoPE)
// Applies rotary embeddings to queries and keys for positional information
// Formula: For each pair (2i, 2i+1):
//   output[2i] = input[2i] * cos(theta) - input[2i+1] * sin(theta)
//   output[2i+1] = input[2i] * sin(theta) + input[2i+1] * cos(theta)
// where theta = pos / (10000^(2i/dim))

fn rope(var q array<f32>, var pos i64, var dim i64) -> array<f32> {
    var result = array<f32>([dim]);
    var i = 0i;

    // Process pairs of dimensions
    while (i < dim) {
        // Compute frequency for this dimension pair
        var freq_exp = (i / 2i) * 2i;  // 2i for pairs
        var freq_scale = freq_exp / dim;

        // theta = pos / (10000^freq_scale)
        // Using approximation: 10000^x ≈ exp(x * ln(10000)) ≈ exp(x * 9.21)
        var theta_scale = 1.0;
        var base = 10000.0;

        // Simple power calculation for small dimensions
        var j = 0i;
        while (j < freq_exp) {
            theta_scale = theta_scale * base;
            j = j + 2i;
        }

        var theta = pos / theta_scale;

        // Compute cos and sin (simplified - will use math functions in actual implementation)
        // For now, using Taylor series approximation
        var cos_theta = 1.0 - (theta * theta / 2.0);
        var sin_theta = theta - (theta * theta * theta / 6.0);

        // Apply rotation
        var q0 = q[i];
        var q1 = q[i + 1i];

        result[i] = q0 * cos_theta - q1 * sin_theta;
        result[i + 1i] = q0 * sin_theta + q1 * cos_theta;

        i = i + 2i;
    }

    return result;
}

fn test_rope() -> f32 {
    var dim = 4i;
    var pos = 0i;

    var q = array<f32>([4]);
    q[0i] = 1.0;
    q[1i] = 0.0;
    q[2i] = 0.0;
    q[3i] = 1.0;

    var result = rope(q, pos, dim);
    return result[0i];
}
