// LLaMA 3.2-1B INT8 Quantized Model
// INT8 weights (pre-transposed), INT8 activations, INT32 accumulation
// 4096 context length, GQA (32 query heads, 8 KV heads)
//
// Architecture:
//   dim = 2048
//   n_layers = 16
//   n_heads = 32
//   n_kv_heads = 8 (GQA)
//   head_dim = 64
//   hidden_dim = 8192
//   vocab_size = 128256
//   max_seq_len = 4096
//   rope_theta = 500000

// Single token forward pass (autoregressive decoding)
fn llama32_1b_forward(
    // Token embedding (FP32 for now, could be INT8)
    f32[] token_embedding,    // [vocab_size, dim] = [128256, 2048]

    // RMSNorm weights (FP32 - small, keep full precision)
    f32[] rms_att_w,          // [n_layers, dim]
    f32[] rms_ffn_w,          // [n_layers, dim]
    f32[] rms_final_w,        // [dim]

    // Attention weights - INT8, PRE-TRANSPOSED for tensor_matmul_nt
    // Original: Wq[dim, dim], stored as Wq_T[dim, dim] (N=dim, K=dim)
    i8[] wq_t,                // [n_layers, dim, dim] pre-transposed
    i8[] wk_t,                // [n_layers, kv_dim, dim] pre-transposed
    i8[] wv_t,                // [n_layers, kv_dim, dim] pre-transposed
    i8[] wo_t,                // [n_layers, dim, dim] pre-transposed

    // FFN weights - INT8, PRE-TRANSPOSED
    i8[] w1_t,                // [n_layers, hidden_dim, dim] gate_proj pre-transposed
    i8[] w2_t,                // [n_layers, dim, hidden_dim] down_proj pre-transposed
    i8[] w3_t,                // [n_layers, hidden_dim, dim] up_proj pre-transposed

    // Output projection (FP32 for final logits precision)
    f32[] wcls,               // [vocab_size, dim]

    // Activation buffers (FP32 for intermediate precision)
    f32[] x,                  // [dim] current activation
    f32[] xb,                 // [dim] after attention RMSNorm
    f32[] xb2,                // [dim] attention output
    f32[] hb,                 // [hidden_dim] FFN hidden
    f32[] hb2,                // [hidden_dim] FFN gate output
    f32[] logits,             // [vocab_size] output logits

    // QKV buffers (INT32 from matmul, will be scaled)
    i32[] q_buf,              // [dim]
    i32[] k_buf,              // [kv_dim]
    i32[] v_buf,              // [kv_dim]

    // Attention scores (FP32 for softmax)
    f32[] att_scores,         // [n_heads, max_seq_len]
    f32[] att_probs,          // [n_heads, max_seq_len]

    // KV Cache - INT8 for memory efficiency
    i8[] k_cache,             // [n_layers, max_seq_len, kv_dim]
    i8[] v_cache,             // [n_layers, max_seq_len, kv_dim]

    // Runtime parameters
    i64 token,                // Current token ID
    i64 pos,                  // Position in sequence

    // Model config
    i64 dim,                  // 2048
    i64 hidden_dim,           // 8192
    i64 n_layers,             // 16
    i64 n_heads,              // 32
    i64 n_kv_heads,           // 8
    i64 vocab_size,           // 128256
    i64 max_seq_len,          // 4096
    i64 head_dim              // 64
) -> f32 {
    var kv_dim = n_kv_heads * head_dim;  // 512
    var n_rep = n_heads / n_kv_heads;     // 4 (GQA factor)
    var seq_len = pos + 1i;

    // ========================================
    // Token Embedding Lookup
    // ========================================
    var emb_offset = token * dim;
    var i = 0i;
    while (i < dim) {
        x[i] = token_embedding[emb_offset + i];
        i = i + 1i;
    }

    // ========================================
    // Transformer Layers
    // ========================================
    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        // ----- Attention RMSNorm -----
        xb = rmsnorm(x, rms_att_w, xb, dim, 0.00001, layer_offset);

        // ----- QKV Projections (INT8 matmul) -----
        // Convert xb to INT8 for matmul (simple quantization)
        // For now, we'll use the scalar loop approach since xb is FP32

        // Q projection: xb @ Wq_T
        var wq_offset = layer * dim * dim;
        i = 0i;
        while (i < dim) {
            var sum = 0;
            var j = 0i;
            while (j < dim) {
                // Quantize xb[j] to INT8 on the fly (scale by 127)
                var xb_i8 = xb[j] * 127.0;
                if (xb_i8 > 127.0) { xb_i8 = 127.0; }
                if (xb_i8 < -128.0) { xb_i8 = -128.0; }
                var w_val = wq_t[wq_offset + i * dim + j] as i32;
                sum = sum + (xb_i8 as i32) * w_val;
                j = j + 1i;
            }
            q_buf[i] = sum;
            i = i + 1i;
        }

        // K projection: xb @ Wk_T
        var wk_offset = layer * kv_dim * dim;
        i = 0i;
        while (i < kv_dim) {
            var sum = 0;
            var j = 0i;
            while (j < dim) {
                var xb_i8 = xb[j] * 127.0;
                if (xb_i8 > 127.0) { xb_i8 = 127.0; }
                if (xb_i8 < -128.0) { xb_i8 = -128.0; }
                var w_val = wk_t[wk_offset + i * dim + j] as i32;
                sum = sum + (xb_i8 as i32) * w_val;
                j = j + 1i;
            }
            k_buf[i] = sum;
            i = i + 1i;
        }

        // V projection: xb @ Wv_T
        var wv_offset = layer * kv_dim * dim;
        i = 0i;
        while (i < kv_dim) {
            var sum = 0;
            var j = 0i;
            while (j < dim) {
                var xb_i8 = xb[j] * 127.0;
                if (xb_i8 > 127.0) { xb_i8 = 127.0; }
                if (xb_i8 < -128.0) { xb_i8 = -128.0; }
                var w_val = wv_t[wv_offset + i * dim + j] as i32;
                sum = sum + (xb_i8 as i32) * w_val;
                j = j + 1i;
            }
            v_buf[i] = sum;
            i = i + 1i;
        }

        // ----- RoPE (on Q and K) -----
        // Apply rotary position embeddings
        var rope_theta = 500000.0;
        i = 0i;
        while (i < dim) {
            var head_dim_i = i % head_dim;
            var freq = 1.0 / pow(rope_theta, (head_dim_i * 1.0) / (head_dim * 1.0));
            var val = (pos * 1.0) * freq;
            var fcr = cos(val);
            var fci = sin(val);

            // Rotate Q (all heads)
            var q0 = q_buf[i] * 1.0;
            var q1 = q_buf[i + 1i] * 1.0;
            q_buf[i] = ((q0 * fcr - q1 * fci) as i32);
            q_buf[i + 1i] = ((q0 * fci + q1 * fcr) as i32);

            // Rotate K (only kv_dim elements)
            if (i < kv_dim) {
                var k0 = k_buf[i] * 1.0;
                var k1 = k_buf[i + 1i] * 1.0;
                k_buf[i] = ((k0 * fcr - k1 * fci) as i32);
                k_buf[i + 1i] = ((k0 * fci + k1 * fcr) as i32);
            }

            i = i + 2i;
        }

        // ----- Store K, V in cache -----
        var cache_offset = layer * max_seq_len * kv_dim + pos * kv_dim;
        i = 0i;
        while (i < kv_dim) {
            // Scale down to INT8 for cache storage
            var k_scaled = k_buf[i] / 16384;  // Scale factor
            var v_scaled = v_buf[i] / 16384;
            k_cache[cache_offset + i] = k_scaled as i8;
            v_cache[cache_offset + i] = v_scaled as i8;
            i = i + 1i;
        }

        // ----- Multi-Head Attention with GQA -----
        var h = 0i;
        while (h < n_heads) {
            var q_offset = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_offset = kv_head * head_dim;

            // Compute attention scores: Q[h] · K_cache[t]
            var t = 0i;
            while (t < seq_len) {
                var score = 0.0;
                var k_pos = layer * max_seq_len * kv_dim + t * kv_dim + kv_head_offset;

                var d = 0i;
                while (d < head_dim) {
                    var q_val = q_buf[q_offset + d] * 1.0;
                    var k_val = k_cache[k_pos + d] * 1.0;
                    score = score + q_val * k_val;
                    d = d + 1i;
                }

                // Scale by 1/sqrt(head_dim)
                score = score / 8.0;  // sqrt(64) = 8
                att_scores[h * max_seq_len + t] = score;
                t = t + 1i;
            }

            // Softmax over attention scores
            var att_offset = h * max_seq_len;
            att_probs = softmax(att_scores, att_probs, seq_len, att_offset, att_offset);

            // Weighted sum of values
            var d = 0i;
            while (d < head_dim) {
                var val = 0.0;
                t = 0i;
                while (t < seq_len) {
                    var v_pos = layer * max_seq_len * kv_dim + t * kv_dim + kv_head_offset;
                    val = val + att_probs[att_offset + t] * (v_cache[v_pos + d] * 1.0);
                    t = t + 1i;
                }
                xb2[q_offset + d] = val;
                d = d + 1i;
            }

            h = h + 1i;
        }

        // ----- Output Projection -----
        var wo_offset = layer * dim * dim;
        i = 0i;
        while (i < dim) {
            var sum = 0.0;
            var j = 0i;
            while (j < dim) {
                var xb2_i8 = xb2[j] * 127.0;
                if (xb2_i8 > 127.0) { xb2_i8 = 127.0; }
                if (xb2_i8 < -128.0) { xb2_i8 = -128.0; }
                var w_val = wo_t[wo_offset + i * dim + j] * 1.0;
                sum = sum + xb2_i8 * w_val;
                j = j + 1i;
            }
            // Residual connection
            x[i] = x[i] + sum / 16384.0;
            i = i + 1i;
        }

        // ----- FFN RMSNorm -----
        xb = rmsnorm(x, rms_ffn_w, xb, dim, 0.00001, layer_offset);

        // ----- SwiGLU FFN -----
        // gate = SiLU(xb @ W1_T)
        // up = xb @ W3_T
        // down = (gate * up) @ W2_T

        var w1_offset = layer * hidden_dim * dim;
        var w3_offset = layer * hidden_dim * dim;

        i = 0i;
        while (i < hidden_dim) {
            // W1 (gate)
            var sum1 = 0.0;
            var j = 0i;
            while (j < dim) {
                var xb_i8 = xb[j] * 127.0;
                if (xb_i8 > 127.0) { xb_i8 = 127.0; }
                if (xb_i8 < -128.0) { xb_i8 = -128.0; }
                sum1 = sum1 + xb_i8 * (w1_t[w1_offset + i * dim + j] * 1.0);
                j = j + 1i;
            }
            // SiLU activation
            var silu_val = sum1 / (1.0 + exp(0.0 - sum1 / 16384.0));
            hb[i] = silu_val;

            // W3 (up)
            var sum3 = 0.0;
            j = 0i;
            while (j < dim) {
                var xb_i8 = xb[j] * 127.0;
                if (xb_i8 > 127.0) { xb_i8 = 127.0; }
                if (xb_i8 < -128.0) { xb_i8 = -128.0; }
                sum3 = sum3 + xb_i8 * (w3_t[w3_offset + i * dim + j] * 1.0);
                j = j + 1i;
            }
            hb2[i] = sum3;

            i = i + 1i;
        }

        // Element-wise multiply gate * up
        i = 0i;
        while (i < hidden_dim) {
            hb[i] = hb[i] * hb2[i] / 16384.0 / 16384.0;
            i = i + 1i;
        }

        // W2 (down projection) + residual
        var w2_offset = layer * dim * hidden_dim;
        i = 0i;
        while (i < dim) {
            var sum = 0.0;
            var j = 0i;
            while (j < hidden_dim) {
                var hb_i8 = hb[j] * 127.0;
                if (hb_i8 > 127.0) { hb_i8 = 127.0; }
                if (hb_i8 < -128.0) { hb_i8 = -128.0; }
                sum = sum + hb_i8 * (w2_t[w2_offset + i * hidden_dim + j] * 1.0);
                j = j + 1i;
            }
            x[i] = x[i] + sum / 16384.0;
            i = i + 1i;
        }

        layer = layer + 1i;
    }

    // ========================================
    // Final RMSNorm
    // ========================================
    xb = rmsnorm(x, rms_final_w, xb, dim, 0.00001, 0i);

    // ========================================
    // Output Classifier (FP32)
    // ========================================
    i = 0i;
    while (i < vocab_size) {
        var sum = 0.0;
        var j = 0i;
        while (j < dim) {
            sum = sum + wcls[i * dim + j] * xb[j];
            j = j + 1i;
        }
        logits[i] = sum;
        i = i + 1i;
    }

    return logits[0];
}
