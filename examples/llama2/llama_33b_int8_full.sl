// LLaMA 33B Fully Quantized Model - NO FP32
// W8A8 matmuls with INT32 accumulation (VNNI)
// INT16/INT32 for all element-wise operations
//
// Architecture:
//   dim = 6144, n_layers = 60, n_heads = 48, n_kv_heads = 16 (GQA=3)
//   head_dim = 128, hidden_dim = 16384, vocab_size = 32000

// ============================================================================
// INT16 RMSNorm
// ============================================================================
fn rmsnorm_i16_33b(
    i16[] x, i16[] weight, i16[] out, i64 dim, i64 offset
) -> i32 {
    var ss = 0i;
    var i = 0i;
    while (i < dim) {
        var val = x[i] as i32;
        ss = ss + (val * val) / 256i;
        i = i + 1i;
    }
    var mean = ss / dim;
    var rsqrt_approx = 16384i;
    if (mean > 256i) { rsqrt_approx = 8192i; }
    if (mean > 1024i) { rsqrt_approx = 4096i; }
    if (mean > 4096i) { rsqrt_approx = 2048i; }
    if (mean > 16384i) { rsqrt_approx = 1024i; }
    i = 0i;
    while (i < dim) {
        var val = x[i] as i32;
        var w = weight[offset + i] as i32;
        var normed = (val * rsqrt_approx) / 256i;
        var result = (normed * w) / 32768i;
        if (result > 32767i) { result = 32767i; }
        if (result < -32768i) { result = -32768i; }
        out[i] = result as i16;
        i = i + 1i;
    }
    return out[0] as i32;
}

// ============================================================================
// INT32 Softmax -> INT8
// ============================================================================
fn softmax_i32_to_i8_33b(
    i32[] scores, i8[] probs, i64 seq_len, i64 in_offset, i64 out_offset
) -> i32 {
    var max_val = scores[in_offset];
    var i = 1i;
    while (i < seq_len) {
        if (scores[in_offset + i] > max_val) { max_val = scores[in_offset + i]; }
        i = i + 1i;
    }
    var sum = 0i;
    i = 0i;
    while (i < seq_len) {
        var diff = scores[in_offset + i] - max_val;
        var e = 128i + diff / 4i;
        if (e < 0i) { e = 0i; }
        if (e > 255i) { e = 255i; }
        probs[out_offset + i] = e as i8;
        sum = sum + e;
        i = i + 1i;
    }
    if (sum > 0i) {
        i = 0i;
        while (i < seq_len) {
            var p = (probs[out_offset + i] as i32) * 127i / sum;
            probs[out_offset + i] = p as i8;
            i = i + 1i;
        }
    }
    return probs[out_offset] as i32;
}

// ============================================================================
// INT32 RoPE (head_dim=128)
// ============================================================================
fn rope_i32_33b(
    i32[] q, i16[] cos_tab, i16[] sin_tab, i32[] out,
    i64 dim, i64 head_dim, i64 pos_offset
) -> i32 {
    var num_heads = dim / head_dim;
    var half_head = head_dim / 2i;
    var h = 0i;
    while (h < num_heads) {
        var head_off = h * head_dim;
        var i = 0i;
        while (i < half_head) {
            var q0 = q[head_off + i * 2i];
            var q1 = q[head_off + i * 2i + 1i];
            var c = cos_tab[pos_offset + i] as i32;
            var s = sin_tab[pos_offset + i] as i32;
            out[head_off + i * 2i] = (q0 * c - q1 * s) / 32768i;
            out[head_off + i * 2i + 1i] = (q0 * s + q1 * c) / 32768i;
            i = i + 1i;
        }
        h = h + 1i;
    }
    return out[0];
}

// ============================================================================
// Decode: Single token forward pass for 33B (GQA=3: n_kv_heads=16, n_heads=48)
// ============================================================================
fn llama_33b_decode_i8(
    i8[] x_i8,                // [dim=6144]
    i16[] rms_att_w,          // [n_layers=60, dim=6144]
    i16[] rms_ffn_w,          // [n_layers=60, dim=6144]
    i16[] rms_final_w,        // [dim=6144]
    i8[] wq_t,                // [n_layers * dim * dim]
    i8[] wk_t,                // [n_layers * kv_dim * dim] kv_dim=2048
    i8[] wv_t,                // [n_layers * kv_dim * dim]
    i8[] wo_t,                // [n_layers * dim * dim]
    i8[] w1_t,                // [n_layers * hidden_dim * dim]
    i8[] w2_t,                // [n_layers * dim * hidden_dim]
    i8[] w3_t,                // [n_layers * hidden_dim * dim]
    i8[] wcls_t,              // [vocab_size=32000, dim=6144]
    i16[] xb_i16,             // [dim=6144]
    i8[] xb_i8,               // [dim=6144]
    i32[] q_i32,              // [dim=6144]
    i32[] k_i32,              // [kv_dim=2048]
    i32[] v_i32,              // [kv_dim=2048]
    i32[] attn_out,           // [dim=6144]
    i8[] ffn_hb,              // [hidden_dim=16384]
    i8[] k_cache,             // [n_layers, max_seq_len, kv_dim]
    i8[] v_cache,             // [n_layers, max_seq_len, kv_dim]
    i32[] att_scores,         // [n_heads=48, max_seq_len]
    i8[] att_probs,           // [n_heads=48, max_seq_len]
    i16[] cos_tab,            // [max_seq_len, head_dim/2=64]
    i16[] sin_tab,            // [max_seq_len, head_dim/2=64]
    i32[] logits,             // [vocab_size=32000]
    i64 pos,
    i64 dim,                  // 6144
    i64 hidden_dim,           // 16384
    i64 n_layers,             // 60
    i64 n_heads,              // 48
    i64 n_kv_heads,           // 16 (GQA=3)
    i64 vocab_size,           // 32000
    i64 max_seq_len,          // 2048
    i64 head_dim              // 128
) -> i32 {
    var kv_dim = n_kv_heads * head_dim;  // 2048
    var n_rep = n_heads / n_kv_heads;     // 3 (GQA ratio)
    var seq_len = pos + 1i;

    // Copy input to INT16
    var i = 0i;
    while (i < dim) {
        xb_i16[i] = (x_i8[i] as i32 * 256i) as i16;
        i = i + 1i;
    }

    // Transformer Layers
    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        // Attention RMSNorm
        var dummy = rmsnorm_i16_33b(xb_i16, rms_att_w, xb_i16, dim, layer_offset);

        // Quantize to INT8
        i = 0i;
        while (i < dim) {
            var val = xb_i16[i] as i32 / 256i;
            if (val > 127i) { val = 127i; }
            if (val < -128i) { val = -128i; }
            xb_i8[i] = val as i8;
            i = i + 1i;
        }

        // QKV Projections
        i8<1, 6144> xb_mat = tensor_from_array(xb_i8, 0i);

        var wq_offset = layer * dim * dim;
        i8<6144, 6144> wq_mat = tensor_from_array(wq_t, wq_offset);
        var Q = tensor_matmul_nt(xb_mat, wq_mat);

        var wk_offset = layer * kv_dim * dim;
        i8<2048, 6144> wk_mat = tensor_from_array(wk_t, wk_offset);
        var K = tensor_matmul_nt(xb_mat, wk_mat);

        var wv_offset = layer * kv_dim * dim;
        i8<2048, 6144> wv_mat = tensor_from_array(wv_t, wv_offset);
        var V = tensor_matmul_nt(xb_mat, wv_mat);

        // Copy to i32 arrays
        i = 0i;
        while (i < dim) {
            q_i32[i] = Q[0i, i];
            i = i + 1i;
        }
        i = 0i;
        while (i < kv_dim) {
            k_i32[i] = K[0i, i];
            v_i32[i] = V[0i, i];
            i = i + 1i;
        }

        // RoPE
        var rope_offset = pos * (head_dim / 2i);
        dummy = rope_i32_33b(q_i32, cos_tab, sin_tab, q_i32, dim, head_dim, rope_offset);
        dummy = rope_i32_33b(k_i32, cos_tab, sin_tab, k_i32, kv_dim, head_dim, rope_offset);

        // Store K, V in cache
        var cache_offset = layer * max_seq_len * kv_dim + pos * kv_dim;
        i = 0i;
        while (i < kv_dim) {
            var k_val = k_i32[i] / 256i;
            var v_val = v_i32[i] / 256i;
            if (k_val > 127i) { k_val = 127i; }
            if (k_val < -128i) { k_val = -128i; }
            if (v_val > 127i) { v_val = 127i; }
            if (v_val < -128i) { v_val = -128i; }
            k_cache[cache_offset + i] = k_val as i8;
            v_cache[cache_offset + i] = v_val as i8;
            i = i + 1i;
        }

        // Multi-Head Attention with GQA
        i = 0i;
        while (i < dim) {
            attn_out[i] = 0i;
            i = i + 1i;
        }

        var h = 0i;
        while (h < n_heads) {
            var head_offset = h * head_dim;
            var kv_head = h / n_rep;  // GQA: map query head to KV head
            var kv_head_offset = kv_head * head_dim;

            var t = 0i;
            while (t < seq_len) {
                var score = 0i;
                var k_pos = layer * max_seq_len * kv_dim + t * kv_dim + kv_head_offset;
                var d = 0i;
                while (d < head_dim) {
                    var q_val = q_i32[head_offset + d];
                    var k_val = k_cache[k_pos + d] as i32;
                    score = score + (q_val * k_val) / 256i;
                    d = d + 1i;
                }
                att_scores[h * max_seq_len + t] = score / 11i;  // 1/sqrt(128)
                t = t + 1i;
            }

            var att_offset = h * max_seq_len;
            dummy = softmax_i32_to_i8_33b(att_scores, att_probs, seq_len, att_offset, att_offset);

            var d = 0i;
            while (d < head_dim) {
                var val = 0i;
                t = 0i;
                while (t < seq_len) {
                    var v_pos = layer * max_seq_len * kv_dim + t * kv_dim + kv_head_offset;
                    var prob = att_probs[att_offset + t] as i32;
                    var v_val = v_cache[v_pos + d] as i32;
                    val = val + prob * v_val;
                    t = t + 1i;
                }
                attn_out[head_offset + d] = val;
                d = d + 1i;
            }
            h = h + 1i;
        }

        // Output Projection
        i = 0i;
        while (i < dim) {
            var val = attn_out[i] / 128i;
            if (val > 127i) { val = 127i; }
            if (val < -128i) { val = -128i; }
            xb_i8[i] = val as i8;
            i = i + 1i;
        }

        i8<1, 6144> attn_mat = tensor_from_array(xb_i8, 0i);
        var wo_offset = layer * dim * dim;
        i8<6144, 6144> wo_mat = tensor_from_array(wo_t, wo_offset);
        var Wo_out = tensor_matmul_nt(attn_mat, wo_mat);

        // Residual
        i = 0i;
        while (i < dim) {
            var res = xb_i16[i] as i32 + Wo_out[0i, i] / 256i;
            if (res > 32767i) { res = 32767i; }
            if (res < -32768i) { res = -32768i; }
            xb_i16[i] = res as i16;
            i = i + 1i;
        }

        // FFN RMSNorm
        dummy = rmsnorm_i16_33b(xb_i16, rms_ffn_w, xb_i16, dim, layer_offset);

        // Quantize for FFN
        i = 0i;
        while (i < dim) {
            var val = xb_i16[i] as i32 / 256i;
            if (val > 127i) { val = 127i; }
            if (val < -128i) { val = -128i; }
            xb_i8[i] = val as i8;
            i = i + 1i;
        }

        // SwiGLU FFN
        var w1_offset = layer * hidden_dim * dim;
        var w3_offset = layer * hidden_dim * dim;
        var w2_offset = layer * dim * hidden_dim;

        i8<1, 6144> ffn_in = tensor_from_array(xb_i8, 0i);
        i8<16384, 6144> w1_mat = tensor_from_array(w1_t, w1_offset);
        var Gate = tensor_matmul_nt(ffn_in, w1_mat);

        i8<16384, 6144> w3_mat = tensor_from_array(w3_t, w3_offset);
        var Up = tensor_matmul_nt(ffn_in, w3_mat);

        var hi = 0i;
        while (hi < hidden_dim) {
            var g = Gate[0i, hi] / 512i;
            var u = Up[0i, hi] / 512i;
            var silu = g / 2i + (g * g) / 2048i;
            var result = (silu * u) / 512i;
            if (result > 127i) { result = 127i; }
            if (result < -128i) { result = -128i; }
            ffn_hb[hi] = result as i8;
            hi = hi + 1i;
        }

        i8<1, 16384> hb_mat = tensor_from_array(ffn_hb, 0i);
        i8<6144, 16384> w2_mat = tensor_from_array(w2_t, w2_offset);
        var Down = tensor_matmul_nt(hb_mat, w2_mat);

        // Residual
        i = 0i;
        while (i < dim) {
            var res = xb_i16[i] as i32 + Down[0i, i] / 256i;
            if (res > 32767i) { res = 32767i; }
            if (res < -32768i) { res = -32768i; }
            xb_i16[i] = res as i16;
            i = i + 1i;
        }

        layer = layer + 1i;
    }

    // Final RMSNorm
    var dummy2 = rmsnorm_i16_33b(xb_i16, rms_final_w, xb_i16, dim, 0i);

    // Quantize final
    i = 0i;
    while (i < dim) {
        var val = xb_i16[i] as i32 / 256i;
        if (val > 127i) { val = 127i; }
        if (val < -128i) { val = -128i; }
        xb_i8[i] = val as i8;
        i = i + 1i;
    }

    // Classifier
    i8<1, 6144> final_mat = tensor_from_array(xb_i8, 0i);
    i8<32000, 6144> wcls_mat = tensor_from_array(wcls_t, 0i);
    var Logits = tensor_matmul_nt(final_mat, wcls_mat);

    i = 0i;
    while (i < vocab_size) {
        logits[i] = Logits[0i, i];
        i = i + 1i;
    }

    return logits[0];
}

// ============================================================================
// Prefill 32 tokens for 33B (GQA=3)
// ============================================================================
fn llama_33b_prefill_32_i8(
    i8[] x_i8,
    i16[] rms_att_w, i16[] rms_ffn_w,
    i8[] wq_t, i8[] wk_t, i8[] wv_t, i8[] wo_t,
    i8[] w1_t, i8[] w2_t, i8[] w3_t,
    i16[] xb_i16, i8[] xb_i8,
    i8[] k_cache, i8[] v_cache,
    i32[] att_scores, i8[] att_probs,
    i32[] attn_out, i8[] ffn_hb,
    i16[] cos_tab, i16[] sin_tab,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers,
    i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> i32 {
    var kv_dim = n_kv_heads * head_dim;  // 2048
    var n_rep = n_heads / n_kv_heads;     // 3

    // Copy input to INT16
    var idx = 0i;
    while (idx < seq_len * dim) {
        xb_i16[idx] = (x_i8[idx] as i32 * 256i) as i16;
        idx = idx + 1i;
    }

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        // RMSNorm per token + quantize
        var t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0i;
            var i = 0i;
            while (i < dim) {
                var val = xb_i16[token_offset + i] as i32;
                ss = ss + (val * val) / 256i;
                i = i + 1i;
            }
            var mean = ss / dim;
            var rsqrt_approx = 16384i;
            if (mean > 256i) { rsqrt_approx = 8192i; }
            if (mean > 1024i) { rsqrt_approx = 4096i; }
            if (mean > 4096i) { rsqrt_approx = 2048i; }
            if (mean > 16384i) { rsqrt_approx = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[token_offset + i] as i32;
                var w = rms_att_w[layer_offset + i] as i32;
                var normed = (val * rsqrt_approx) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 127i) { result = 127i; }
                if (result < -128i) { result = -128i; }
                xb_i8[token_offset + i] = result as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        // QKV Projections (batched)
        i8<32, 6144> xb_mat = tensor_from_array(xb_i8, 0i);

        var wq_offset = layer * dim * dim;
        i8<6144, 6144> wq_mat = tensor_from_array(wq_t, wq_offset);
        var Q = tensor_matmul_nt(xb_mat, wq_mat);

        var wk_offset = layer * kv_dim * dim;
        i8<2048, 6144> wk_mat = tensor_from_array(wk_t, wk_offset);
        var K = tensor_matmul_nt(xb_mat, wk_mat);

        var wv_offset = layer * kv_dim * dim;
        i8<2048, 6144> wv_mat = tensor_from_array(wv_t, wv_offset);
        var V = tensor_matmul_nt(xb_mat, wv_mat);

        // Store KV cache
        var cache_layer_offset = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < kv_dim) {
                var k_val = K[t, i] / 256i;
                var v_val = V[t, i] / 256i;
                if (k_val > 127i) { k_val = 127i; }
                if (k_val < -128i) { k_val = -128i; }
                if (v_val > 127i) { v_val = 127i; }
                if (v_val < -128i) { v_val = -128i; }
                k_cache[cache_layer_offset + t * kv_dim + i] = k_val as i8;
                v_cache[cache_layer_offset + t * kv_dim + i] = v_val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        // Causal Attention with GQA
        var h = 0i;
        while (h < n_heads) {
            var head_offset = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_offset = kv_head * head_dim;

            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0i;
                    var d = 0i;
                    while (d < head_dim) {
                        score = score + Q[q_pos, head_offset + d] * (k_cache[cache_layer_offset + k_pos * kv_dim + kv_head_offset + d] as i32);
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 11i;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -1000000i;
                    k_pos = k_pos + 1i;
                }

                // Softmax
                var att_offset = h * seq_len * seq_len + q_pos * seq_len;
                var max_val = att_scores[att_offset];
                var ki = 1i;
                while (ki <= q_pos) {
                    if (att_scores[att_offset + ki] > max_val) { max_val = att_scores[att_offset + ki]; }
                    ki = ki + 1i;
                }
                var sum = 0i;
                ki = 0i;
                while (ki <= q_pos) {
                    var diff = att_scores[att_offset + ki] - max_val;
                    var e = 128i + diff / 4i;
                    if (e < 0i) { e = 0i; }
                    if (e > 255i) { e = 255i; }
                    att_probs[att_offset + ki] = e as i8;
                    sum = sum + e;
                    ki = ki + 1i;
                }
                if (sum > 0i) {
                    ki = 0i;
                    while (ki <= q_pos) {
                        var p = (att_probs[att_offset + ki] as i32) * 127i / sum;
                        att_probs[att_offset + ki] = p as i8;
                        ki = ki + 1i;
                    }
                }

                // Weighted sum of V
                var d = 0i;
                while (d < head_dim) {
                    var val = 0i;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        val = val + (att_probs[att_offset + vi] as i32) * (v_cache[cache_layer_offset + vi * kv_dim + kv_head_offset + d] as i32);
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + head_offset + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        // Output projection
        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                var val = attn_out[t * dim + i] / 128i;
                if (val > 127i) { val = 127i; }
                if (val < -128i) { val = -128i; }
                xb_i8[t * dim + i] = val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<32, 6144> attn_mat = tensor_from_array(xb_i8, 0i);
        var wo_offset = layer * dim * dim;
        i8<6144, 6144> wo_mat = tensor_from_array(wo_t, wo_offset);
        var Wo_out = tensor_matmul_nt(attn_mat, wo_mat);

        // Residual
        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                var res = xb_i16[t * dim + i] as i32 + Wo_out[t, i] / 256i;
                if (res > 32767i) { res = 32767i; }
                if (res < -32768i) { res = -32768i; }
                xb_i16[t * dim + i] = res as i16;
                i = i + 1i;
            }
            t = t + 1i;
        }

        // FFN RMSNorm + quantize
        t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0i;
            var i = 0i;
            while (i < dim) {
                var val = xb_i16[token_offset + i] as i32;
                ss = ss + (val * val) / 256i;
                i = i + 1i;
            }
            var mean = ss / dim;
            var rsqrt_approx = 16384i;
            if (mean > 256i) { rsqrt_approx = 8192i; }
            if (mean > 1024i) { rsqrt_approx = 4096i; }
            if (mean > 4096i) { rsqrt_approx = 2048i; }
            if (mean > 16384i) { rsqrt_approx = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[token_offset + i] as i32;
                var w = rms_ffn_w[layer_offset + i] as i32;
                var normed = (val * rsqrt_approx) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 127i) { result = 127i; }
                if (result < -128i) { result = -128i; }
                xb_i8[token_offset + i] = result as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        // SwiGLU FFN
        var w1_offset = layer * hidden_dim * dim;
        var w3_offset = layer * hidden_dim * dim;
        var w2_offset = layer * dim * hidden_dim;

        i8<32, 6144> ffn_in = tensor_from_array(xb_i8, 0i);
        i8<16384, 6144> w1_mat = tensor_from_array(w1_t, w1_offset);
        var Gate = tensor_matmul_nt(ffn_in, w1_mat);

        i8<16384, 6144> w3_mat = tensor_from_array(w3_t, w3_offset);
        var Up = tensor_matmul_nt(ffn_in, w3_mat);

        t = 0i;
        while (t < seq_len) {
            var hi = 0i;
            while (hi < hidden_dim) {
                var g = Gate[t, hi] / 512i;
                var u = Up[t, hi] / 512i;
                var silu = g / 2i + (g * g) / 2048i;
                var result = (silu * u) / 512i;
                if (result > 127i) { result = 127i; }
                if (result < -128i) { result = -128i; }
                ffn_hb[t * hidden_dim + hi] = result as i8;
                hi = hi + 1i;
            }
            t = t + 1i;
        }

        i8<32, 16384> hb_mat = tensor_from_array(ffn_hb, 0i);
        i8<6144, 16384> w2_mat = tensor_from_array(w2_t, w2_offset);
        var Down = tensor_matmul_nt(hb_mat, w2_mat);

        // Residual
        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                var res = xb_i16[t * dim + i] as i32 + Down[t, i] / 256i;
                if (res > 32767i) { res = 32767i; }
                if (res < -32768i) { res = -32768i; }
                xb_i16[t * dim + i] = res as i16;
                i = i + 1i;
            }
            t = t + 1i;
        }

        layer = layer + 1i;
    }

    return xb_i16[(seq_len - 1i) * dim] as i32;
}
