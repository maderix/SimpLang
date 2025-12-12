// Element-wise Operations Benchmark (INT16/INT32)
// Tests remaining FP32 loops for INT conversion
//
// 1. Residual add: x += proj_out (INT32 + INT32)
// 2. SiLU * up combined (INT32 inputs)
// 3. Scale/quantize (INT32 -> INT8)

// ============================================================
// Residual Add: x[i] = x[i] + y[i] (both INT32)
// Simulates: x_fp32[i] = x_fp32[i] + Wo_out[0i, i] / 65536.0
// In INT: x_i32[i] = x_i32[i] + y_i32[i] (accumulate in INT32)
// ============================================================

fn residual_add_2048(
    i32[] x,      // [2048] accumulator (Q16 format)
    i32[] y,      // [2048] projection output
    i32[] out     // [2048] result
) -> i32 {
    var i = 0i;
    while (i < 2048i) {
        out[i] = x[i] + y[i];
        i = i + 1i;
    }
    return out[0];
}

// ============================================================
// SiLU * Up Combined (INT32 -> INT8)
// gate and up are INT32 from matmul, output INT8
// silu(x) ≈ x/2 + x²/2048 (quadratic approx)
// result = silu(gate) * up, scaled to INT8
// ============================================================

fn silu_mul_8192(
    i32[] gate,   // [8192] gate projection (INT32)
    i32[] up,     // [8192] up projection (INT32)
    i8[] out      // [8192] result (INT8)
) -> i32 {
    var i = 0i;
    while (i < 8192i) {
        // Scale down gate from INT32 matmul output
        var g = gate[i] / 256;  // Scale to reasonable range

        // SiLU approx: g/2 + g²/2048
        var silu = g / 2 + (g * g) / 2048;

        // Scale down up
        var u = up[i] / 256;

        // Multiply and scale to INT8
        var result = (silu * u) / 256;

        // Clamp to INT8
        if (result > 127) { result = 127; }
        if (result < -128) { result = -128; }

        out[i] = result as i8;
        i = i + 1i;
    }
    return out[0] as i32;
}

// ============================================================
// Quantize INT32 -> INT8 with scaling
// Simulates: val = attn_out[i] / 256.0; clamp; cast to i8
// ============================================================

fn quantize_i32_i8_2048(
    i32[] x,      // [2048] input INT32
    i8[] out,     // [2048] output INT8
    i64 scale     // division factor
) -> i32 {
    var i = 0i;
    while (i < 2048i) {
        var val = x[i] / scale;
        if (val > 127i) { val = 127i; }
        if (val < -128i) { val = -128i; }
        out[i] = val as i8;
        i = i + 1i;
    }
    return out[0] as i32;
}

// ============================================================
// RoPE Integer Approximation
// Uses precomputed sin/cos tables (INT16)
// q_rot = q0 * cos - q1 * sin, q1_rot = q0 * sin + q1 * cos
// ============================================================

fn rope_i32_64(
    i32[] q,          // [64] query vector (INT32)
    i16[] cos_table,  // [32] precomputed cos (Q15)
    i16[] sin_table,  // [32] precomputed sin (Q15)
    i32[] out         // [64] rotated output
) -> i32 {
    var i = 0i;
    while (i < 32i) {
        var q0 = q[i * 2i];
        var q1 = q[i * 2i + 1i];
        var c = cos_table[i] as i32;
        var s = sin_table[i] as i32;

        // Q15 multiply: (a * b) >> 15
        var q0_rot = (q0 * c - q1 * s) / 32768;
        var q1_rot = (q0 * s + q1 * c) / 32768;

        out[i * 2i] = q0_rot;
        out[i * 2i + 1i] = q1_rot;
        i = i + 1i;
    }
    return out[0];
}

// Larger version for full dim
fn rope_i32_2048(
    i32[] q,
    i16[] cos_table,  // [1024] for 2048/2 pairs
    i16[] sin_table,
    i32[] out
) -> i32 {
    var i = 0i;
    while (i < 1024i) {
        var q0 = q[i * 2i];
        var q1 = q[i * 2i + 1i];
        var c = cos_table[i] as i32;
        var s = sin_table[i] as i32;

        var q0_rot = (q0 * c - q1 * s) / 32768;
        var q1_rot = (q0 * s + q1 * c) / 32768;

        out[i * 2i] = q0_rot;
        out[i * 2i + 1i] = q1_rot;
        i = i + 1i;
    }
    return out[0];
}
