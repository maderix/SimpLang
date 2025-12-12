// INT8 Attention Benchmark - Q·K and Attn·V only
// Tests tensor_matmul for VNNI optimization
//
// Q·K: Q[1, head_dim] @ K[seq_len, head_dim]^T -> scores[1, seq_len]
// Attn·V: probs[1, seq_len] @ V[seq_len, head_dim] -> out[1, head_dim]
//
// Fixed seq_len sizes: 128, 512, 2048

// ============================================================
// Q·K Attention Scores using tensor_matmul_nt
// ============================================================

// Q[1, 64] @ K[128, 64]^T -> scores[1, 128]
fn attention_qk_128(
    i8[] q,           // [64] query vector
    i8[] k_cache,     // [128 * 64] key cache
    i32[] scores      // [128] output scores
) -> i32 {
    i8<1, 64> q_mat = tensor_from_array(q, 0i);
    i8<128, 64> k_mat = tensor_from_array(k_cache, 0i);

    var result = tensor_matmul_nt(q_mat, k_mat);  // [1, 128] i32

    // Copy to output array
    var i = 0i;
    while (i < 128i) {
        scores[i] = result[0i, i];
        i = i + 1i;
    }
    return scores[0];
}

// Q[1, 64] @ K[512, 64]^T -> scores[1, 512]
fn attention_qk_512(
    i8[] q,
    i8[] k_cache,
    i32[] scores
) -> i32 {
    i8<1, 64> q_mat = tensor_from_array(q, 0i);
    i8<512, 64> k_mat = tensor_from_array(k_cache, 0i);

    var result = tensor_matmul_nt(q_mat, k_mat);

    var i = 0i;
    while (i < 512i) {
        scores[i] = result[0i, i];
        i = i + 1i;
    }
    return scores[0];
}

// Q[1, 64] @ K[2048, 64]^T -> scores[1, 2048]
fn attention_qk_2048(
    i8[] q,
    i8[] k_cache,
    i32[] scores
) -> i32 {
    i8<1, 64> q_mat = tensor_from_array(q, 0i);
    i8<2048, 64> k_mat = tensor_from_array(k_cache, 0i);

    var result = tensor_matmul_nt(q_mat, k_mat);

    var i = 0i;
    while (i < 2048i) {
        scores[i] = result[0i, i];
        i = i + 1i;
    }
    return scores[0];
}

// ============================================================
// Attn·V Weighted Sum using tensor_matmul
// probs[1, seq_len] @ V[seq_len, 64] -> out[1, 64]
// ============================================================

// probs[1, 128] @ V[128, 64] -> out[1, 64]
fn attention_av_128(
    i8[] probs,       // [128] attention probabilities (scaled to INT8)
    i8[] v_cache,     // [128 * 64] value cache
    i32[] out         // [64] output
) -> i32 {
    i8<1, 128> p_mat = tensor_from_array(probs, 0i);
    i8<128, 64> v_mat = tensor_from_array(v_cache, 0i);

    var result = tensor_matmul(p_mat, v_mat);  // [1, 64] i32

    var i = 0i;
    while (i < 64i) {
        out[i] = result[0i, i];
        i = i + 1i;
    }
    return out[0];
}

// probs[1, 512] @ V[512, 64] -> out[1, 64]
fn attention_av_512(
    i8[] probs,
    i8[] v_cache,
    i32[] out
) -> i32 {
    i8<1, 512> p_mat = tensor_from_array(probs, 0i);
    i8<512, 64> v_mat = tensor_from_array(v_cache, 0i);

    var result = tensor_matmul(p_mat, v_mat);

    var i = 0i;
    while (i < 64i) {
        out[i] = result[0i, i];
        i = i + 1i;
    }
    return out[0];
}

// probs[1, 2048] @ V[2048, 64] -> out[1, 64]
fn attention_av_2048(
    i8[] probs,
    i8[] v_cache,
    i32[] out
) -> i32 {
    i8<1, 2048> p_mat = tensor_from_array(probs, 0i);
    i8<2048, 64> v_mat = tensor_from_array(v_cache, 0i);

    var result = tensor_matmul(p_mat, v_mat);

    var i = 0i;
    while (i < 64i) {
        out[i] = result[0i, i];
        i = i + 1i;
    }
    return out[0];
}
