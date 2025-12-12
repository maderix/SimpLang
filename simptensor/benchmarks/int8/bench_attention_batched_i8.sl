// INT8 Batched Attention Benchmark
// Batch 4 heads together for I=4 VNNI tiling
//
// Q·K: Q[4, head_dim] @ K[seq_len, head_dim]^T -> scores[4, seq_len]
// Attn·V: probs[4, seq_len] @ V[seq_len, head_dim] -> out[4, head_dim]

// ============================================================
// Batched Q·K: 4 heads at once for I=4 tiling
// ============================================================

// Q[4, 64] @ K[128, 64]^T -> scores[4, 128]
fn attention_qk_4h_128(
    i8[] q,           // [4 * 64] = 4 query vectors
    i8[] k_cache,     // [128 * 64] key cache (shared across heads)
    i32[] scores      // [4 * 128] output scores
) -> i32 {
    i8<4, 64> q_mat = tensor_from_array(q, 0i);
    i8<128, 64> k_mat = tensor_from_array(k_cache, 0i);

    var result = tensor_matmul_nt(q_mat, k_mat);  // [4, 128] i32

    // Copy to output array
    var i = 0i;
    while (i < 4i) {
        var j = 0i;
        while (j < 128i) {
            scores[i * 128i + j] = result[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return scores[0];
}

// Q[4, 64] @ K[512, 64]^T -> scores[4, 512]
fn attention_qk_4h_512(
    i8[] q,
    i8[] k_cache,
    i32[] scores
) -> i32 {
    i8<4, 64> q_mat = tensor_from_array(q, 0i);
    i8<512, 64> k_mat = tensor_from_array(k_cache, 0i);

    var result = tensor_matmul_nt(q_mat, k_mat);

    var i = 0i;
    while (i < 4i) {
        var j = 0i;
        while (j < 512i) {
            scores[i * 512i + j] = result[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return scores[0];
}

// Q[4, 64] @ K[2048, 64]^T -> scores[4, 2048]
fn attention_qk_4h_2048(
    i8[] q,
    i8[] k_cache,
    i32[] scores
) -> i32 {
    i8<4, 64> q_mat = tensor_from_array(q, 0i);
    i8<2048, 64> k_mat = tensor_from_array(k_cache, 0i);

    var result = tensor_matmul_nt(q_mat, k_mat);

    var i = 0i;
    while (i < 4i) {
        var j = 0i;
        while (j < 2048i) {
            scores[i * 2048i + j] = result[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return scores[0];
}

// ============================================================
// Batched Attn·V: 4 heads at once
// ============================================================

// probs[4, 128] @ V[128, 64] -> out[4, 64]
fn attention_av_4h_128(
    i8[] probs,       // [4 * 128]
    i8[] v_cache,     // [128 * 64]
    i32[] out         // [4 * 64]
) -> i32 {
    i8<4, 128> p_mat = tensor_from_array(probs, 0i);
    i8<128, 64> v_mat = tensor_from_array(v_cache, 0i);

    var result = tensor_matmul(p_mat, v_mat);  // [4, 64] i32

    var i = 0i;
    while (i < 4i) {
        var j = 0i;
        while (j < 64i) {
            out[i * 64i + j] = result[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return out[0];
}

// probs[4, 512] @ V[512, 64] -> out[4, 64]
fn attention_av_4h_512(
    i8[] probs,
    i8[] v_cache,
    i32[] out
) -> i32 {
    i8<4, 512> p_mat = tensor_from_array(probs, 0i);
    i8<512, 64> v_mat = tensor_from_array(v_cache, 0i);

    var result = tensor_matmul(p_mat, v_mat);

    var i = 0i;
    while (i < 4i) {
        var j = 0i;
        while (j < 64i) {
            out[i * 64i + j] = result[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return out[0];
}

// probs[4, 2048] @ V[2048, 64] -> out[4, 64]
fn attention_av_4h_2048(
    i8[] probs,
    i8[] v_cache,
    i32[] out
) -> i32 {
    i8<4, 2048> p_mat = tensor_from_array(probs, 0i);
    i8<2048, 64> v_mat = tensor_from_array(v_cache, 0i);

    var result = tensor_matmul(p_mat, v_mat);

    var i = 0i;
    while (i < 4i) {
        var j = 0i;
        while (j < 64i) {
            out[i * 64i + j] = result[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    return out[0];
}
