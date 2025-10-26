// Quantized LLaMA 2 transformer - W4 quantized weights, FP32 activations
// All weight matrices stored in 4-bit format with per-group scales/zeros

// Quantized operations are now MLIR builtins (dequant_w4 and matmul_quant)
// Defined in SimpOps.td and lowered in ConvertSimpToMemRef.cpp

fn llama2_quant_forward(
    // FP32 arrays
    f32[] token_emb,
    f32[] rms_att_w,
    f32[] rms_ffn_w,
    f32[] rms_final_w,
    f32[] wcls,

    // Quantized weight arrays (i8)
    i8[] qweights,      // All quantized weights concatenated
    f32[] scales,       // Scales for all weights
    f32[] zeros,        // Zero points for all weights

    // Activation buffers
    f32[] x,
    f32[] xb,
    f32[] xb2,
    f32[] q,
    f32[] k,
    f32[] v,
    f32[] att,
    f32[] att_soft,
    f32[] hb,
    f32[] hb_silu,
    f32[] hb2,
    f32[] logits,
    f32[] key_cache,
    f32[] value_cache,

    // Model config
    i64 token,
    i64 pos,
    i64 dim,
    i64 hidden_dim,
    i64 n_layers,
    i64 n_heads,
    i64 n_kv_heads,
    i64 vocab_size,
    i64 seq_len,
    i64 group_size,

    // Weight offsets in qweights array
    i64 wq_off,
    i64 wk_off,
    i64 wv_off,
    i64 wo_off,
    i64 w1_off,
    i64 w2_off,
    i64 w3_off
) -> f32 {
    var kv_dim = (dim * n_kv_heads) / n_heads;
    var head_size = dim / n_heads;

    // Token embedding (FP32)
    var token_offset = token * dim;
    var emb_idx = 0i;
    while (emb_idx < dim) {
        x[emb_idx] = token_emb[token_offset + emb_idx];
        emb_idx = emb_idx + 1i;
    }

    // Transformer layers
    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        // RMSNorm for attention
        xb = rmsnorm(x, rms_att_w, xb, dim, 0.00001, layer_offset);

        // QKV projections (quantized)
        var wq_layer_off = wq_off + layer * dim * dim;
        var wk_layer_off = wk_off + layer * dim * kv_dim;
        var wv_layer_off = wv_off + layer * dim * kv_dim;

        q = matmul_quant(qweights, scales, zeros, xb, q, 1536i, 1536i, 128i, wq_layer_off);
        k = matmul_quant(qweights, scales, zeros, xb, k, 1536i, 1536i, 128i, wk_layer_off);
        v = matmul_quant(qweights, scales, zeros, xb, v, 1536i, 1536i, 128i, wv_layer_off);

        // Store KV cache
        var kv_pos_off = layer * seq_len * kv_dim + pos * kv_dim;
        var kv_idx = 0i;
        while (kv_idx < kv_dim) {
            key_cache[kv_pos_off + kv_idx] = k[kv_idx];
            value_cache[kv_pos_off + kv_idx] = v[kv_idx];
            kv_idx = kv_idx + 1i;
        }

        // Multi-head attention
        var h = 0i;
        while (h < n_heads) {
            var q_off = h * head_size;
            var kv_head = (h * n_kv_heads) / n_heads;
            var kv_off = kv_head * head_size;

            // Compute attention scores
            var t = 0i;
            while (t <= pos) {
                var score = 0.0;
                var key_pos_off = layer * seq_len * kv_dim + t * kv_dim;

                var hs = 0i;
                while (hs < head_size) {
                    score = score + q[q_off + hs] * key_cache[key_pos_off + kv_off + hs];
                    hs = hs + 1i;
                }

                score = score / sqrt(head_size + 0.0);
                att[t] = score;
                t = t + 1i;
            }

            // Softmax
            var att_len = pos + 1i;
            att_soft = softmax(att, att_soft, att_len, 0i, 0i);

            // Weighted sum of values
            var hs2 = 0i;
            while (hs2 < head_size) {
                var val = 0.0;
                var t2 = 0i;
                while (t2 <= pos) {
                    var val_pos_off = layer * seq_len * kv_dim + t2 * kv_dim;
                    val = val + att_soft[t2] * value_cache[val_pos_off + kv_off + hs2];
                    t2 = t2 + 1i;
                }
                xb2[q_off + hs2] = val;
                hs2 = hs2 + 1i;
            }

            h = h + 1i;
        }

        // Output projection (quantized)
        var wo_layer_off = wo_off + layer * dim * dim;
        xb = matmul_quant(qweights, scales, zeros, xb2, xb, 1536i, 1536i, 128i, wo_layer_off);

        // Residual
        var res_idx = 0i;
        while (res_idx < dim) {
            x[res_idx] = x[res_idx] + xb[res_idx];
            res_idx = res_idx + 1i;
        }

        // RMSNorm for FFN
        xb = rmsnorm(x, rms_ffn_w, xb, dim, 0.00001, layer_offset);

        // SwiGLU FFN (quantized)
        var w1_layer_off = w1_off + layer * hidden_dim * dim;
        var w3_layer_off = w3_off + layer * hidden_dim * dim;
        var w2_layer_off = w2_off + layer * dim * hidden_dim;

        hb = matmul_quant(qweights, scales, zeros, xb, hb, 6144i, 1536i, 128i, w1_layer_off);
        hb2 = matmul_quant(qweights, scales, zeros, xb, hb2, 6144i, 1536i, 128i, w3_layer_off);

        hb_silu = silu(hb, hb_silu, hidden_dim);

        var mul_idx = 0i;
        while (mul_idx < hidden_dim) {
            hb[mul_idx] = hb_silu[mul_idx] * hb2[mul_idx];
            mul_idx = mul_idx + 1i;
        }

        xb = matmul_quant(qweights, scales, zeros, hb, xb, 1536i, 6144i, 128i, w2_layer_off);

        // Residual
        var res_idx2 = 0i;
        while (res_idx2 < dim) {
            x[res_idx2] = x[res_idx2] + xb[res_idx2];
            res_idx2 = res_idx2 + 1i;
        }

        layer = layer + 1i;
    }

    // Final RMSNorm
    xb = rmsnorm(x, rms_final_w, xb, dim, 0.00001, 0i);

    // Classifier (FP32)
    var c_idx = 0i;
    while (c_idx < vocab_size) {
        var sum = 0.0;
        var c_j = 0i;
        while (c_j < dim) {
            sum = sum + wcls[c_idx * dim + c_j] * xb[c_j];
            c_j = c_j + 1i;
        }
        logits[c_idx] = sum;
        c_idx = c_idx + 1i;
    }

    return logits[0];
}
