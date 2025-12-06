// Prefill Matmul Benchmark
// Tests tensor_matmul_nt with M > 1 (batch token processing)
// For LLaMA 3.2-1B: dim=2048, hidden_dim=8192

// Prefill QKV projection: [M, 2048] @ [2048, 2048]_nt -> [M, 2048]
// M = 64 tokens
fn prefill_qkv_64(
    i8[] input,      // [64, 2048]
    i8[] weight,     // [2048, 2048] pre-transposed
    i32[] output     // [64, 2048]
) -> i32 {
    i8<64, 2048> A = tensor_from_array(input, 0i);
    i8<2048, 2048> B = tensor_from_array(weight, 0i);
    var C = tensor_matmul_nt(A, B);  // [64, 2048]

    // Store result
    var i = 0i;
    while (i < 64i) {
        var j = 0i;
        while (j < 2048i) {
            output[i * 2048i + j] = C[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0i, 0i];
}

// M = 128 tokens
fn prefill_qkv_128(
    i8[] input,      // [128, 2048]
    i8[] weight,     // [2048, 2048] pre-transposed
    i32[] output     // [128, 2048]
) -> i32 {
    i8<128, 2048> A = tensor_from_array(input, 0i);
    i8<2048, 2048> B = tensor_from_array(weight, 0i);
    var C = tensor_matmul_nt(A, B);  // [128, 2048]

    var i = 0i;
    while (i < 128i) {
        var j = 0i;
        while (j < 2048i) {
            output[i * 2048i + j] = C[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0i, 0i];
}

// M = 256 tokens
fn prefill_qkv_256(
    i8[] input,      // [256, 2048]
    i8[] weight,     // [2048, 2048] pre-transposed
    i32[] output     // [256, 2048]
) -> i32 {
    i8<256, 2048> A = tensor_from_array(input, 0i);
    i8<2048, 2048> B = tensor_from_array(weight, 0i);
    var C = tensor_matmul_nt(A, B);  // [256, 2048]

    var i = 0i;
    while (i < 256i) {
        var j = 0i;
        while (j < 2048i) {
            output[i * 2048i + j] = C[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0i, 0i];
}

// M = 512 tokens
fn prefill_qkv_512(
    i8[] input,      // [512, 2048]
    i8[] weight,     // [2048, 2048] pre-transposed
    i32[] output     // [512, 2048]
) -> i32 {
    i8<512, 2048> A = tensor_from_array(input, 0i);
    i8<2048, 2048> B = tensor_from_array(weight, 0i);
    var C = tensor_matmul_nt(A, B);  // [512, 2048]

    var i = 0i;
    while (i < 512i) {
        var j = 0i;
        while (j < 2048i) {
            output[i * 2048i + j] = C[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0i, 0i];
}

// Prefill FFN gate/up: [M, 2048] @ [8192, 2048]_nt -> [M, 8192]
// M = 64
fn prefill_ffn_up_64(
    i8[] input,      // [64, 2048]
    i8[] weight,     // [8192, 2048] pre-transposed
    i32[] output     // [64, 8192]
) -> i32 {
    i8<64, 2048> A = tensor_from_array(input, 0i);
    i8<8192, 2048> B = tensor_from_array(weight, 0i);
    var C = tensor_matmul_nt(A, B);  // [64, 8192]

    var i = 0i;
    while (i < 64i) {
        var j = 0i;
        while (j < 8192i) {
            output[i * 8192i + j] = C[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0i, 0i];
}

// M = 128
fn prefill_ffn_up_128(
    i8[] input,      // [128, 2048]
    i8[] weight,     // [8192, 2048] pre-transposed
    i32[] output     // [128, 8192]
) -> i32 {
    i8<128, 2048> A = tensor_from_array(input, 0i);
    i8<8192, 2048> B = tensor_from_array(weight, 0i);
    var C = tensor_matmul_nt(A, B);  // [128, 8192]

    var i = 0i;
    while (i < 128i) {
        var j = 0i;
        while (j < 8192i) {
            output[i * 8192i + j] = C[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0i, 0i];
}

// Prefill FFN down: [M, 8192] @ [2048, 8192]_nt -> [M, 2048]
// M = 64
fn prefill_ffn_down_64(
    i8[] input,      // [64, 8192]
    i8[] weight,     // [2048, 8192] pre-transposed
    i32[] output     // [64, 2048]
) -> i32 {
    i8<64, 8192> A = tensor_from_array(input, 0i);
    i8<2048, 8192> B = tensor_from_array(weight, 0i);
    var C = tensor_matmul_nt(A, B);  // [64, 2048]

    var i = 0i;
    while (i < 64i) {
        var j = 0i;
        while (j < 2048i) {
            output[i * 2048i + j] = C[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0i, 0i];
}

// M = 128
fn prefill_ffn_down_128(
    i8[] input,      // [128, 8192]
    i8[] weight,     // [2048, 8192] pre-transposed
    i32[] output     // [128, 2048]
) -> i32 {
    i8<128, 8192> A = tensor_from_array(input, 0i);
    i8<2048, 8192> B = tensor_from_array(weight, 0i);
    var C = tensor_matmul_nt(A, B);  // [128, 2048]

    var i = 0i;
    while (i < 128i) {
        var j = 0i;
        while (j < 2048i) {
            output[i * 2048i + j] = C[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return C[0i, 0i];
}
