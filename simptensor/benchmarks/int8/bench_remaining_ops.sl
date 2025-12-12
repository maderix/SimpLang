// Remaining Ops Benchmark - INT8/INT16/INT32
// 1. RoPE with precomputed INT16 sin/cos tables
// 2. Residual add (INT32)
// 3. FFN SiLU*up (INT32 -> INT8)

// ============================================================
// RoPE: Rotary Position Embedding (INT32 with INT16 tables)
// q_rot[2i] = q[2i]*cos - q[2i+1]*sin
// q_rot[2i+1] = q[2i]*sin + q[2i+1]*cos
// cos/sin tables are Q15 format (32767 = 1.0)
// ============================================================

fn rope_i32_64(
    i32[] q,          // [64] query (INT32 from matmul)
    i16[] cos_tab,    // [32] precomputed cos (Q15)
    i16[] sin_tab,    // [32] precomputed sin (Q15)
    i32[] out         // [64] rotated output
) -> i32 {
    var i = 0i;
    while (i < 32i) {
        var q0 = q[i * 2i];
        var q1 = q[i * 2i + 1i];
        var c = cos_tab[i] as i32;
        var s = sin_tab[i] as i32;

        // Q15 multiply: result = (a * b) / 32768
        out[i * 2i] = (q0 * c - q1 * s) / 32768;
        out[i * 2i + 1i] = (q0 * s + q1 * c) / 32768;
        i = i + 1i;
    }
    return out[0];
}

fn rope_i32_2048(
    i32[] q,          // [2048] full Q vector
    i16[] cos_tab,    // [1024]
    i16[] sin_tab,
    i32[] out
) -> i32 {
    var i = 0i;
    while (i < 1024i) {
        var q0 = q[i * 2i];
        var q1 = q[i * 2i + 1i];
        var c = cos_tab[i] as i32;
        var s = sin_tab[i] as i32;

        out[i * 2i] = (q0 * c - q1 * s) / 32768;
        out[i * 2i + 1i] = (q0 * s + q1 * c) / 32768;
        i = i + 1i;
    }
    return out[0];
}

// ============================================================
// Residual Add: out = x + y (INT32)
// ============================================================

fn residual_add_2048(
    i32[] x,
    i32[] y,
    i32[] out
) -> i32 {
    var i = 0i;
    while (i < 2048i) {
        out[i] = x[i] + y[i];
        i = i + 1i;
    }
    return out[0];
}

// ============================================================
// FFN SiLU*Up: silu(gate) * up -> INT8
// gate, up are INT32 from matmul
// silu(x) ≈ x/2 + x²/2048
// Output scaled to INT8
// Using if-else with array-store pattern for clamping
// ============================================================

fn ffn_silu_mul_8192_v2(
    i32[] gate,
    i32[] up,
    i8[] out,
    i32[] tmp         // temp buffer for clamping
) -> i32 {
    var i = 0i;
    while (i < 8192i) {
        var g = gate[i] / 512i;
        var u = up[i] / 512i;

        var silu = g / 2i + (g * g) / 2048i;
        var result = (silu * u) / 512i;

        // Clamp using if-else with array store
        tmp[i] = result;
        if (result > 127i) {
            tmp[i] = 127i;
        } else {
            if (result < -128i) {
                tmp[i] = -128i;
            }
        }

        out[i] = tmp[i] as i8;
        i = i + 1i;
    }
    return out[0] as i32;
}
