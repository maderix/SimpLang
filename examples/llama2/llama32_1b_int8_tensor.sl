// LLaMA 3.2-1B INT8 Quantized Model - Tensor Operations Version
// Uses tensor_matmul_nt for VNNI-optimized INT8 matmuls
// INT8 weights (pre-transposed), INT8 activations, INT32 accumulation
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

// Single token forward pass with tensor ops for projections
fn llama32_forward(
    // Input: already quantized to INT8
    i8[] x_i8,                // [dim] = [2048] quantized input

    // RMSNorm weights (FP32 - keep precision for normalization)
    f32[] rms_att_w,          // [n_layers, dim]
    f32[] rms_ffn_w,          // [n_layers, dim]
    f32[] rms_final_w,        // [dim]

    // Attention weights - INT8, PRE-TRANSPOSED [N, K] layout
    i8[] wq_t,                // [n_layers * dim * dim]
    i8[] wk_t,                // [n_layers * kv_dim * dim]
    i8[] wv_t,                // [n_layers * kv_dim * dim]
    i8[] wo_t,                // [n_layers * dim * dim]

    // FFN weights - INT8, PRE-TRANSPOSED
    i8[] w1_t,                // [n_layers * hidden_dim * dim]
    i8[] w2_t,                // [n_layers * dim * hidden_dim]
    i8[] w3_t,                // [n_layers * hidden_dim * dim]

    // Classifier (keep FP32 for output precision)
    f32[] wcls,               // [vocab_size, dim]

    // FP32 buffers for normalization
    f32[] x_fp32,             // [dim]
    f32[] xb_fp32,            // [dim]
    f32[] logits,             // [vocab_size]

    // INT8 buffer for normalized input
    i8[] xb_i8,               // [dim]

    // KV Cache - INT8
    i8[] k_cache,             // [n_layers, max_seq_len, kv_dim]
    i8[] v_cache,             // [n_layers, max_seq_len, kv_dim]

    // Attention scores (FP32 for softmax precision)
    f32[] att_scores,         // [n_heads, max_seq_len]
    f32[] att_probs,          // [n_heads, max_seq_len]

    // Attention output buffer
    i32[] attn_out,           // [dim]

    // FFN hidden buffer (INT8)
    i8[] ffn_hb,              // [hidden_dim] for gate*up result

    // Runtime
    i64 pos,

    // Config
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
    var n_rep = n_heads / n_kv_heads;     // 4
    var seq_len = pos + 1i;

    // Copy input to FP32 for normalization
    var i = 0i;
    while (i < dim) {
        x_fp32[i] = x_i8[i] * 1.0;
        i = i + 1i;
    }

    // ========================================
    // Transformer Layers
    // ========================================
    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        // ----- Attention RMSNorm -> xb_fp32 -----
        xb_fp32 = rmsnorm(x_fp32, rms_att_w, xb_fp32, dim, 0.00001, layer_offset);

        // Quantize normalized input to INT8
        i = 0i;
        while (i < dim) {
            var val = xb_fp32[i] * 127.0;
            if (val > 127.0) { val = 127.0; }
            if (val < -128.0) { val = -128.0; }
            xb_i8[i] = val as i8;
            i = i + 1i;
        }

        // ----- QKV Projections using tensor_matmul_nt -----
        // Q: [1, 2048] @ [2048, 2048]_nt -> [1, 2048]
        i8<1, 2048> xb_mat = tensor_from_array(xb_i8, 0i);

        var wq_offset = layer * dim * dim;
        i8<2048, 2048> wq_mat = tensor_from_array(wq_t, wq_offset);
        var Q = tensor_matmul_nt(xb_mat, wq_mat);  // [1, 2048] i32

        // K: [1, 2048] @ [512, 2048]_nt -> [1, 512]
        var wk_offset = layer * kv_dim * dim;
        i8<512, 2048> wk_mat = tensor_from_array(wk_t, wk_offset);
        var K = tensor_matmul_nt(xb_mat, wk_mat);  // [1, 512] i32

        // V: [1, 2048] @ [512, 2048]_nt -> [1, 512]
        var wv_offset = layer * kv_dim * dim;
        i8<512, 2048> wv_mat = tensor_from_array(wv_t, wv_offset);
        var V = tensor_matmul_nt(xb_mat, wv_mat);  // [1, 512] i32

        // ----- RoPE on Q and K -----
        var rope_theta = 500000.0;
        i = 0i;
        while (i < dim) {
            var head_dim_i = i % head_dim;
            var freq = 1.0 / pow(rope_theta, (head_dim_i * 1.0) / (head_dim * 1.0));
            var val = (pos * 1.0) * freq;
            var fcr = cos(val);
            var fci = sin(val);

            // Rotate Q
            var q0 = Q[0i, i] * 1.0;
            var q1 = Q[0i, i + 1i] * 1.0;
            // Note: tensor element assignment not directly supported,
            // would need to copy back. For now skip RoPE in tensor version.

            i = i + 2i;
        }

        // ----- Store K, V in cache (scale to INT8) -----
        var cache_offset = layer * max_seq_len * kv_dim + pos * kv_dim;
        i = 0i;
        while (i < kv_dim) {
            // Scale down i32 to i8 range
            var k_val = K[0i, i] / 256;
            var v_val = V[0i, i] / 256;
            k_cache[cache_offset + i] = k_val as i8;
            v_cache[cache_offset + i] = v_val as i8;
            i = i + 1i;
        }

        // ----- Multi-Head Attention with GQA -----
        // Clear attention output
        i = 0i;
        while (i < dim) {
            attn_out[i] = 0;
            i = i + 1i;
        }

        var h = 0i;
        while (h < n_heads) {
            var q_offset = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_offset = kv_head * head_dim;

            // Compute attention scores
            var t = 0i;
            while (t < seq_len) {
                var score = 0.0;
                var k_pos = layer * max_seq_len * kv_dim + t * kv_dim + kv_head_offset;

                var d = 0i;
                while (d < head_dim) {
                    var q_val = Q[0i, q_offset + d] * 1.0;
                    var k_val = k_cache[k_pos + d] * 1.0;
                    score = score + q_val * k_val;
                    d = d + 1i;
                }
                score = score / 8.0;
                att_scores[h * max_seq_len + t] = score;
                t = t + 1i;
            }

            // Softmax
            var att_offset = h * max_seq_len;
            att_probs = softmax(att_scores, att_probs, seq_len, att_offset, att_offset);

            // Weighted sum
            var d = 0i;
            while (d < head_dim) {
                var val = 0.0;
                t = 0i;
                while (t < seq_len) {
                    var v_pos = layer * max_seq_len * kv_dim + t * kv_dim + kv_head_offset;
                    val = val + att_probs[att_offset + t] * (v_cache[v_pos + d] * 1.0);
                    t = t + 1i;
                }
                attn_out[q_offset + d] = val as i32;
                d = d + 1i;
            }

            h = h + 1i;
        }

        // ----- Output Projection -----
        // Quantize attention output to INT8
        i = 0i;
        while (i < dim) {
            var val = attn_out[i] * 1.0 / 256.0;
            if (val > 127.0) { val = 127.0; }
            if (val < -128.0) { val = -128.0; }
            xb_i8[i] = val as i8;
            i = i + 1i;
        }

        // Wo: [1, 2048] @ [2048, 2048]_nt -> [1, 2048]
        i8<1, 2048> attn_mat = tensor_from_array(xb_i8, 0i);
        var wo_offset = layer * dim * dim;
        i8<2048, 2048> wo_mat = tensor_from_array(wo_t, wo_offset);
        var Wo_out = tensor_matmul_nt(attn_mat, wo_mat);  // [1, 2048] i32

        // Residual connection
        i = 0i;
        while (i < dim) {
            x_fp32[i] = x_fp32[i] + Wo_out[0i, i] * 1.0 / 65536.0;
            i = i + 1i;
        }

        // ----- FFN RMSNorm -----
        xb_fp32 = rmsnorm(x_fp32, rms_ffn_w, xb_fp32, dim, 0.00001, layer_offset);

        // Quantize for FFN
        i = 0i;
        while (i < dim) {
            var val = xb_fp32[i] * 127.0;
            if (val > 127.0) { val = 127.0; }
            if (val < -128.0) { val = -128.0; }
            xb_i8[i] = val as i8;
            i = i + 1i;
        }

        // ----- SwiGLU FFN with tensor_matmul_nt -----
        // gate = SiLU(xb @ W1_T), up = xb @ W3_T, out = (gate * up) @ W2_T
        var w1_offset = layer * hidden_dim * dim;
        var w3_offset = layer * hidden_dim * dim;
        var w2_offset = layer * dim * hidden_dim;

        // W1 (gate projection): [1, 2048] @ [8192, 2048]_nt -> [1, 8192]
        i8<8192, 2048> w1_mat = tensor_from_array(w1_t, w1_offset);
        var Gate = tensor_matmul_nt(xb_mat, w1_mat);  // [1, 8192] i32

        // W3 (up projection): [1, 2048] @ [8192, 2048]_nt -> [1, 8192]
        i8<8192, 2048> w3_mat = tensor_from_array(w3_t, w3_offset);
        var Up = tensor_matmul_nt(xb_mat, w3_mat);  // [1, 8192] i32

        // SiLU(gate) * up -> ffn_hb (quantize to INT8)
        var hi = 0i;
        while (hi < hidden_dim) {
            var gate_f = Gate[0i, hi] * 1.0 / 65536.0;
            var silu = gate_f / (1.0 + exp(0.0 - gate_f));
            var up_f = Up[0i, hi] * 1.0 / 65536.0;
            var hb_val = silu * up_f * 127.0;
            ffn_hb[hi] = hb_val as i8;
            hi = hi + 1i;
        }

        // W2 (down projection): [1, 8192] @ [2048, 8192]_nt -> [1, 2048]
        i8<1, 8192> hb_mat = tensor_from_array(ffn_hb, 0i);
        i8<2048, 8192> w2_mat = tensor_from_array(w2_t, w2_offset);
        var Down = tensor_matmul_nt(hb_mat, w2_mat);  // [1, 2048] i32

        // Residual connection
        i = 0i;
        while (i < dim) {
            x_fp32[i] = x_fp32[i] + Down[0i, i] * 1.0 / 65536.0;
            i = i + 1i;
        }

        layer = layer + 1i;
    }

    // ========================================
    // Final RMSNorm
    // ========================================
    xb_fp32 = rmsnorm(x_fp32, rms_final_w, xb_fp32, dim, 0.00001, 0i);

    // ========================================
    // Classifier
    // ========================================
    i = 0i;
    while (i < vocab_size) {
        var sum = 0.0;
        var j = 0i;
        while (j < dim) {
            sum = sum + wcls[i * dim + j] * xb_fp32[j];
            j = j + 1i;
        }
        logits[i] = sum;
        i = i + 1i;
    }

    return logits[0];
}
