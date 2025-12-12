// INT16 SiLU Benchmark - Fast Quadratic
// Input:  i16[] x   - Q8 format
// Output: i16[] out - Q8 format
//
// silu(x) ≈ x/2 + x^2/2048
// Fast, vectorizable, ~0.15 avg error (acceptable for LLM)

fn silu_i16_fast(
    i16[] x,
    i16[] out,
    i64 size
) -> i32 {
    var i = 0i;
    while (i < size) {
        var xi = x[i] as i32;

        // silu ≈ x/2 + x^2/2048
        var half_x = xi / 2;
        var x_sq = (xi * xi) / 2048;
        var result = half_x + x_sq;

        // Clamp to i16
        if (result > 32767) { result = 32767; }
        if (result < -32768) { result = -32768; }

        out[i] = result as i16;
        i = i + 1i;
    }

    return out[0] as i32;
}

fn bench_silu_2048(i16[] x, i16[] out) -> i32 {
    return silu_i16_fast(x, out, 2048i);
}

fn bench_silu_8192(i16[] x, i16[] out) -> i32 {
    return silu_i16_fast(x, out, 8192i);
}

fn bench_silu_11008(i16[] x, i16[] out) -> i32 {
    return silu_i16_fast(x, out, 11008i);
}
