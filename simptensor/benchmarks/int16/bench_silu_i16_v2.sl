// INT16 SiLU Benchmark - Improved Piecewise Sigmoid
// Uses array-store pattern to avoid MLIR type mismatch in if-else
//
// silu(x) = x * sigmoid(x)
// Improved approximation with proper saturation:
//   x < -1024 (-4.0): sigmoid ≈ 0, silu ≈ 0
//   x > 1024 (+4.0):  sigmoid ≈ 1, silu ≈ x
//   else: sigmoid ≈ 0.5 + x/8 (linear approx in [-4,4])

fn silu_i16_piecewise(
    i16[] x,
    i16[] out,
    i32[] tmp,    // temp buffer for avoiding if-var-assign
    i64 size
) -> i32 {
    var i = 0i;
    while (i < size) {
        var xi = x[i] as i32;

        // Default: middle region sigmoid ≈ 128 + x/8
        var sigmoid_q8 = 128 + xi / 8;
        tmp[i] = (xi * sigmoid_q8) / 256;

        // Saturation: x < -1024 → silu ≈ 0
        if (xi < -1024) {
            tmp[i] = 0;
        }

        // Saturation: x > 1024 → silu ≈ x
        if (xi > 1024) {
            tmp[i] = xi;
        }

        // Clamp to i16 range
        var result = tmp[i];
        tmp[i] = result;
        if (result > 32767) {
            tmp[i] = 32767;
        }
        if (result < -32768) {
            tmp[i] = -32768;
        }

        out[i] = tmp[i] as i16;
        i = i + 1i;
    }

    return out[0] as i32;
}

// Cubic correction version: sigmoid ≈ 0.5 + x/4 ± x²/8192
fn silu_i16_cubic(
    i16[] x,
    i16[] out,
    i32[] tmp,
    i64 size
) -> i32 {
    var i = 0i;
    while (i < size) {
        var xi = x[i] as i32;

        // Base sigmoid: 128 + x/4
        var sig = 128 + xi / 4;

        // Quadratic correction (bends curve at edges)
        var x_sq = (xi * xi) / 8192;
        tmp[i] = sig;
        if (xi > 0) {
            tmp[i] = sig - x_sq;
        }
        if (xi < 0) {
            tmp[i] = sig + x_sq;
        }
        sig = tmp[i];

        // Clamp sigmoid to [0, 256]
        if (sig < 0) {
            tmp[i] = 0;
        }
        if (sig > 256) {
            tmp[i] = 256;
        }
        sig = tmp[i];

        // Compute silu = x * sigmoid / 256
        var result = (xi * sig) / 256;

        // Saturation regions
        tmp[i] = result;
        if (xi < -1024) {
            tmp[i] = 0;
        }
        if (xi > 1024) {
            tmp[i] = xi;
        }
        result = tmp[i];

        // Clamp to i16
        if (result > 32767) {
            tmp[i] = 32767;
        }
        if (result < -32768) {
            tmp[i] = -32768;
        }

        out[i] = tmp[i] as i16;
        i = i + 1i;
    }

    return out[0] as i32;
}

fn bench_silu_v2_2048(i16[] x, i16[] out, i32[] tmp) -> i32 {
    return silu_i16_piecewise(x, out, tmp, 2048i);
}

fn bench_silu_v2_8192(i16[] x, i16[] out, i32[] tmp) -> i32 {
    return silu_i16_piecewise(x, out, tmp, 8192i);
}

fn bench_silu_cubic_2048(i16[] x, i16[] out, i32[] tmp) -> i32 {
    return silu_i16_cubic(x, out, tmp, 2048i);
}

fn bench_silu_cubic_8192(i16[] x, i16[] out, i32[] tmp) -> i32 {
    return silu_i16_cubic(x, out, tmp, 8192i);
}
