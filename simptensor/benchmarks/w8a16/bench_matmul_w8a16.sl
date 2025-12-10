// W8A16 MatMul Benchmark - Hi-Lo Split for VNNI
// Activations: INT16 (Q8 format)
// Weights: INT8
// Accumulator: INT32
// Output: INT32 (apply scale externally)
//
// Hi-Lo Split Algorithm:
//   INT16_act = (hi << 8) | lo
//   INT8_wt × INT16_act = (INT8_wt × hi) << 8 + (INT8_wt × lo)
//   Both multiplies use INT8×INT8 (VNNI compatible)

// INT8 matmul kernel - VNNI-friendly pattern
// This should trigger VNNI optimization
fn matmul_i8_nt(
    i8[] A,           // [M * K] INT8
    i8[] W_T,         // [N * K] INT8 (transposed)
    i32[] C,          // [M * N] INT32 output
    i64 M, i64 N, i64 K
) -> i32 {
    var i = 0i;
    while (i < M) {
        var j = 0i;
        while (j < N) {
            var acc = 0;
            var k = 0i;
            while (k < K) {
                var a = A[i * K + k] as i32;
                var w = W_T[j * K + k] as i32;
                acc = acc + a * w;
                k = k + 1i;
            }
            C[i * N + j] = acc;
            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0];
}

// W8A16 MatMul with Hi-Lo split - VNNI version
// Uses separate matmul calls for hi and lo paths
fn matmul_w8a16_hilo(
    i16[] A,           // [M * K] activations INT16
    i8[] W_T,          // [N * K] weights INT8 (transposed)
    i32[] C,           // [M * N] output INT32
    i8[] A_hi,         // [M * K] temp buffer for high bytes
    i8[] A_lo,         // [M * K] temp buffer for low bytes (unsigned)
    i64 M, i64 N, i64 K
) -> i32 {
    // Step 1: Split A into hi/lo bytes
    var idx = 0i;
    var total = M * K;
    while (idx < total) {
        var a = A[idx];
        // Hi-Lo split using bitwise ops
        A_hi[idx] = (a >> 8) as i8;
        // For lo, we want unsigned 0-255 but store as i8
        // The i8 representation is fine, we'll handle sign in combine
        A_lo[idx] = (a & 255) as i8;
        idx = idx + 1i;
    }

    // Step 2: Compute two INT8 matmuls (VNNI-eligible)
    var i = 0i;
    while (i < M) {
        var j = 0i;
        while (j < N) {
            var acc_hi = 0;
            var acc_lo = 0;
            var k = 0i;
            while (k < K) {
                var w = W_T[j * K + k] as i32;
                var hi = A_hi[i * K + k] as i32;
                // CRITICAL: lo must be unsigned (0-255), mask off sign extension
                var lo = (A_lo[i * K + k] as i32) & 255;

                acc_hi = acc_hi + hi * w;
                acc_lo = acc_lo + lo * w;
                k = k + 1i;
            }

            // Combine: (hi << 8) + lo
            C[i * N + j] = (acc_hi << 8) + acc_lo;

            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0];
}

// W8A16 with explicit separate matmul passes (VNNI-friendly)
// This version does two complete INT8 matmuls to maximize VNNI usage
fn matmul_w8a16_two_pass(
    i16[] A,           // [M * K] activations INT16
    i8[] W_T,          // [N * K] weights INT8 (transposed)
    i32[] C,           // [M * N] output INT32
    i8[] A_hi,         // [M * K] temp buffer for high bytes
    i8[] A_lo,         // [M * K] temp buffer for low bytes
    i32[] C_hi,        // [M * N] temp for hi result
    i32[] C_lo,        // [M * N] temp for lo result
    i64 M, i64 N, i64 K
) -> i32 {
    // Step 1: Split A into hi/lo bytes
    var idx = 0i;
    var total = M * K;
    while (idx < total) {
        var a = A[idx];
        A_hi[idx] = (a >> 8) as i8;
        A_lo[idx] = (a & 255) as i8;
        idx = idx + 1i;
    }

    // Step 2a: Hi matmul (VNNI eligible)
    var i = 0i;
    while (i < M) {
        var j = 0i;
        while (j < N) {
            var acc = 0;
            var k = 0i;
            while (k < K) {
                var a = A_hi[i * K + k] as i32;
                var w = W_T[j * K + k] as i32;
                acc = acc + a * w;
                k = k + 1i;
            }
            C_hi[i * N + j] = acc;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Step 2b: Lo matmul (need unsigned handling)
    i = 0i;
    while (i < M) {
        var j = 0i;
        while (j < N) {
            var acc = 0;
            var k = 0i;
            while (k < K) {
                // Mask to unsigned for lo byte
                var a = (A_lo[i * K + k] as i32) & 255;
                var w = W_T[j * K + k] as i32;
                acc = acc + a * w;
                k = k + 1i;
            }
            C_lo[i * N + j] = acc;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Step 3: Combine results
    idx = 0i;
    var out_total = M * N;
    while (idx < out_total) {
        C[idx] = (C_hi[idx] << 8) + C_lo[idx];
        idx = idx + 1i;
    }

    return C[0];
}

// W8A16 MatMul with per-channel scales
// Output = ((hi << 8) + lo) * scale[j] >> 16
fn matmul_w8a16_scaled(
    i16[] A,           // [M * K] activations INT16
    i8[] W_T,          // [N * K] weights INT8 (transposed)
    i16[] C,           // [M * N] output INT16 (scaled)
    i8[] A_hi,         // [M * K] temp buffer
    i8[] A_lo,         // [M * K] temp buffer
    i32[] scales,      // [N] per-channel scales (Q16)
    i64 M, i64 N, i64 K
) -> i32 {
    // Split A into hi/lo
    var idx = 0i;
    var total = M * K;
    while (idx < total) {
        var a = A[idx];
        A_hi[idx] = (a >> 8) as i8;
        A_lo[idx] = (a & 255) as i8;
        idx = idx + 1i;
    }

    // Compute with scaling
    var i = 0i;
    while (i < M) {
        var j = 0i;
        while (j < N) {
            var acc_hi = 0;
            var acc_lo = 0;
            var k = 0i;
            while (k < K) {
                var w = W_T[j * K + k] as i32;
                var hi = A_hi[i * K + k] as i32;
                // CRITICAL: lo must be unsigned (0-255), mask off sign extension
                var lo = (A_lo[i * K + k] as i32) & 255;

                acc_hi = acc_hi + hi * w;
                acc_lo = acc_lo + lo * w;
                k = k + 1i;
            }

            // Combine and scale
            var raw = (acc_hi << 8) + acc_lo;
            var scaled = (raw * scales[j]) >> 16;

            // Clamp to INT16
            if (scaled > 32767) { scaled = 32767; }
            if (scaled < -32768) { scaled = -32768; }
            C[i * N + j] = scaled as i16;

            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0] as i32;
}

// Reference: Direct INT16 x INT8 matmul (for correctness check)
fn matmul_i16xi8_ref(
    i16[] A,           // [M * K] INT16
    i8[] W_T,          // [N * K] INT8
    i32[] C,           // [M * N] INT32
    i64 M, i64 N, i64 K
) -> i32 {
    var i = 0i;
    while (i < M) {
        var j = 0i;
        while (j < N) {
            var acc = 0;
            var k = 0i;
            while (k < K) {
                var a = A[i * K + k] as i32;
                var w = W_T[j * K + k] as i32;
                acc = acc + a * w;
                k = k + 1i;
            }
            C[i * N + j] = acc;
            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0];
}

// Benchmark sizes
fn bench_w8a16_64x64(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo) -> i32 {
    return matmul_w8a16_hilo(A, W_T, C, A_hi, A_lo, 64i, 64i, 64i);
}

fn bench_w8a16_128x128(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo) -> i32 {
    return matmul_w8a16_hilo(A, W_T, C, A_hi, A_lo, 128i, 128i, 128i);
}

fn bench_w8a16_256x256(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo) -> i32 {
    return matmul_w8a16_hilo(A, W_T, C, A_hi, A_lo, 256i, 256i, 256i);
}

fn bench_w8a16_512x512(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo) -> i32 {
    return matmul_w8a16_hilo(A, W_T, C, A_hi, A_lo, 512i, 512i, 512i);
}

fn bench_w8a16_1024x1024(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo) -> i32 {
    return matmul_w8a16_hilo(A, W_T, C, A_hi, A_lo, 1024i, 1024i, 1024i);
}

fn bench_w8a16_2048x2048(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo) -> i32 {
    return matmul_w8a16_hilo(A, W_T, C, A_hi, A_lo, 2048i, 2048i, 2048i);
}

// Reference benchmarks
fn bench_ref_64x64(i16[] A, i8[] W_T, i32[] C) -> i32 {
    return matmul_i16xi8_ref(A, W_T, C, 64i, 64i, 64i);
}

fn bench_ref_256x256(i16[] A, i8[] W_T, i32[] C) -> i32 {
    return matmul_i16xi8_ref(A, W_T, C, 256i, 256i, 256i);
}

// LLaMA-like sizes: batch x dim x dim
fn bench_w8a16_32x2048x2048(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo) -> i32 {
    return matmul_w8a16_hilo(A, W_T, C, A_hi, A_lo, 32i, 2048i, 2048i);
}

fn bench_w8a16_64x2048x2048(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo) -> i32 {
    return matmul_w8a16_hilo(A, W_T, C, A_hi, A_lo, 64i, 2048i, 2048i);
}

fn bench_w8a16_128x2048x2048(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo) -> i32 {
    return matmul_w8a16_hilo(A, W_T, C, A_hi, A_lo, 128i, 2048i, 2048i);
}

// Two-pass benchmarks (VNNI-optimized structure)
fn bench_w8a16_2pass_256x256(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo, i32[] C_hi, i32[] C_lo) -> i32 {
    return matmul_w8a16_two_pass(A, W_T, C, A_hi, A_lo, C_hi, C_lo, 256i, 256i, 256i);
}

fn bench_w8a16_2pass_512x512(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo, i32[] C_hi, i32[] C_lo) -> i32 {
    return matmul_w8a16_two_pass(A, W_T, C, A_hi, A_lo, C_hi, C_lo, 512i, 512i, 512i);
}

fn bench_w8a16_2pass_1024x1024(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo, i32[] C_hi, i32[] C_lo) -> i32 {
    return matmul_w8a16_two_pass(A, W_T, C, A_hi, A_lo, C_hi, C_lo, 1024i, 1024i, 1024i);
}

fn bench_w8a16_2pass_2048x2048(i16[] A, i8[] W_T, i32[] C, i8[] A_hi, i8[] A_lo, i32[] C_hi, i32[] C_lo) -> i32 {
    return matmul_w8a16_two_pass(A, W_T, C, A_hi, A_lo, C_hi, C_lo, 2048i, 2048i, 2048i);
}

