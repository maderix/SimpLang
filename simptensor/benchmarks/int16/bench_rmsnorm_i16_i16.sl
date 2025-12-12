// INT16 RMSNorm Benchmark
// Input:  i16[] x      - Q8 format (8 fractional bits)
// Input:  i16[] weight - Q8 format
// Output: i16[] out    - Q8 format
//
// Fixed-point Q8: 256 = 1.0, range -128.0 to 127.996
//
// Algorithm:
//   ss = mean(x^2)
//   rsqrt_ss = 1/sqrt(ss + eps)  [Newton-Raphson, 4 iterations]
//   out[i] = x[i] * rsqrt_ss * weight[i]

// INT16 RMSNorm - data passed from host
fn rmsnorm_i16(
    i16[] x,        // Q8 input
    i16[] weight,   // Q8 weights
    i16[] out,      // Q8 output
    i64 dim
) -> i32 {
    // Step 1: Compute mean of squares (i32 accumulator)
    // x is Q8, x^2 is Q16, sum is Q16
    var ss = 0;
    var i = 0i;
    while (i < dim) {
        var xi = x[i] as i32;
        ss = ss + (xi * xi);
        i = i + 1i;
    }
    // Mean: still Q16
    ss = ss / dim;

    // Add epsilon in Q16 (eps ~ 0.00001 * 65536 ~ 1)
    ss = ss + 1;

    // Step 2: Newton-Raphson rsqrt (4 iterations)
    // Output in Q8 format
    var y = 256;  // Initial guess: 1.0 in Q8

    var j = 0i;
    while (j < 4i) {
        var y2 = (y * y);           // Q16
        var xy2 = (ss * y2) / 65536; // Q16
        var term = 196608 - xy2;     // 3.0 in Q16 - xy2
        y = (y * term) / 131072;     // Q8
        j = j + 1i;
    }

    // Step 3: Apply normalization
    // out = x * rsqrt * weight (all Q8)
    i = 0i;
    while (i < dim) {
        var xi = x[i] as i32;
        var wi = weight[i] as i32;

        var norm = (xi * y) / 256;       // Q8 * Q8 / 256 = Q8
        var result = (norm * wi) / 256;  // Q8 * Q8 / 256 = Q8

        // Clamp to i16
        if (result > 32767) { result = 32767; }
        if (result < -32768) { result = -32768; }

        out[i] = result as i16;
        i = i + 1i;
    }

    return out[0] as i32;
}

// dim=2048 (LLaMA attention)
fn bench_rmsnorm_2048(
    i16[] x,
    i16[] weight,
    i16[] out
) -> i32 {
    return rmsnorm_i16(x, weight, out, 2048i);
}

// dim=8192 (LLaMA FFN)
fn bench_rmsnorm_8192(
    i16[] x,
    i16[] weight,
    i16[] out
) -> i32 {
    return rmsnorm_i16(x, weight, out, 8192i);
}

// dim=4096 (LLaMA-7B)
fn bench_rmsnorm_4096(
    i16[] x,
    i16[] weight,
    i16[] out
) -> i32 {
    return rmsnorm_i16(x, weight, out, 4096i);
}
