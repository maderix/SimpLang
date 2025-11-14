// LLaMA 2 3B Model
// Config: dim=2048, n_layers=32, n_heads=32, vocab=32000, seq_len=512
// ~3B parameters

fn llama2_forward(
    // Token embeddings
    f32[] token_embedding_table,

    // Transformer weights (24 layers)
    f32[] rms_att_w,  // [n_layers, dim]
    f32[] wq, f32[] wk, f32[] wv, f32[] wo,  // attention weights
    f32[] rms_ffn_w,  // [n_layers, dim]
    f32[] w1, f32[] w2, f32[] w3,  // FFN weights

    // Final layer
    f32[] rms_final_w,  // [dim]
    f32[] wcls,  // classifier weights [vocab_size, dim]

    // Activations and state
    f32[] x,        // input activation [dim]
    f32[] xb,       // intermediate buffer [dim]
    f32[] xb2,      // second buffer [dim]
    f32[] hb,       // hidden dimension buffer [hidden_dim=8192]
    f32[] hb_silu,  // SiLU activation buffer [hidden_dim]
    f32[] q, f32[] k, f32[] v,  // query, key, value [dim]
    f32[] att,      // attention scores [n_heads, seq_len]
    f32[] att_soft, // softmax attention [n_heads, seq_len]
    f32[] logits,   // output logits [vocab_size]

    // KV cache [n_layers, seq_len, dim]
    f32[] key_cache,
    f32[] value_cache,

    // Hyperparameters
    i64 token,      // current token
    i64 pos,        // position in sequence
    i64 dim,        // 1536
    i64 hidden_dim, // 6144 (4 * dim)
    i64 n_layers,   // 24
    i64 n_heads,    // 16
    i64 n_kv_heads, // 16
    i64 vocab_size, // 32000
    i64 seq_len     // 512
) -> f32 {
    // Hyperparameters
    var head_size = dim / n_heads;  // 64
    var kv_dim = (dim * n_kv_heads) / n_heads;  // 2048

    // Copy token embedding into x
    var i = 0i;
    while (i < dim) {
        x[i] = token_embedding_table[token * dim + i];
        i = i + 1i;
    }

    // Forward through all layers
    var layer = 0i;
    while (layer < n_layers) {
        // Attention RMSNorm
        var rms_w_offset = layer * dim;
        xb = rmsnorm(x, rms_att_w, xb, dim, 0.00001, rms_w_offset);

        // QKV projections using matmul builtin
        var qkv_offset = layer * dim * dim;

        // Zero output buffers (matmul does accumulation C += A*B)
        i = 0i;
        while (i < dim) {
            q[i] = 0.0;
            k[i] = 0.0;
            v[i] = 0.0;
            i = i + 1i;
        }

        // Q = wq @ xb (treating xb as column vector [dim x 1])
        // matmul(lhs, rhs, output, m, k, n, lhs_offset, rhs_offset, output_offset)
        q = matmul(wq, xb, q, 2048i, 2048i, 1i, qkv_offset, 0i, 0i);

        // K = wk @ xb
        k = matmul(wk, xb, k, 2048i, 2048i, 1i, qkv_offset, 0i, 0i);

        // V = wv @ xb
        v = matmul(wv, xb, v, 2048i, 2048i, 1i, qkv_offset, 0i, 0i);

        // RoPE: Rotary Position Embedding on Q and K
        i = 0i;
        while (i < dim) {
            var head_dim_i = i % head_size;
            var head_dim_f = head_dim_i * 1.0;  // Convert to float
            var head_size_f = head_size * 1.0;
            var freq = 1.0 / pow(10000.0, head_dim_f / head_size_f);
            var pos_f = pos * 1.0;
            var val = pos_f * freq;
            var fcr = cos(val);
            var fci = sin(val);

            // Always rotate Q
            var q0 = q[i];
            var q1 = q[i + 1i];
            q[i] = q0 * fcr - q1 * fci;
            q[i + 1i] = q0 * fci + q1 * fcr;

            // Only rotate K if i < kv_dim (for multi-query attention)
            if (i < kv_dim) {
                var k0 = k[i];
                var k1 = k[i + 1i];
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1i] = k0 * fci + k1 * fcr;
            }

            i = i + 2i;
        }

        // Store KV in cache
        var kv_offset = layer * seq_len * dim + pos * dim;
        i = 0i;
        while (i < dim) {
            key_cache[kv_offset + i] = k[i];
            value_cache[kv_offset + i] = v[i];
            i = i + 1i;
        }

        // Multi-head attention
        var h = 0i;
        while (h < n_heads) {
            var q_offset = h * head_size;

            // Attention scores for this head
            var t = 0i;
            var att_len = pos + 1i;
            while (t < att_len) {
                var score = 0.0;
                var kv_pos = layer * seq_len * dim + t * dim + h * head_size;

                i = 0i;
                while (i < head_size) {
                    score = score + q[q_offset + i] * key_cache[kv_pos + i];
                    i = i + 1i;
                }
                score = score / 8.0;  // sqrt(head_size=96) ≈ 9.75
                att[h * seq_len + t] = score;
                t = t + 1i;
            }

            // Softmax (with offset for current head)
            var att_offset = h * seq_len;
            att_soft = softmax(att, att_soft, att_len, att_offset, att_offset);

            // Weighted sum of values
            i = 0i;
            while (i < head_size) {
                var val = 0.0;
                t = 0i;
                while (t < att_len) {
                    var v_pos = layer * seq_len * dim + t * dim + h * head_size;
                    val = val + att_soft[h * seq_len + t] * value_cache[v_pos + i];
                    t = t + 1i;
                }
                xb[q_offset + i] = val;
                i = i + 1i;
            }

            h = h + 1i;
        }

        // Output projection: xb2 = wo @ xb
        var wo_offset = layer * dim * dim;
        // Zero output buffer
        i = 0i;
        while (i < dim) {
            xb2[i] = 0.0;
            i = i + 1i;
        }
        xb2 = matmul(wo, xb, xb2, 2048i, 2048i, 1i, wo_offset, 0i, 0i);

        // Residual connection
        i = 0i;
        while (i < dim) {
            x[i] = x[i] + xb2[i];
            i = i + 1i;
        }

        // FFN RMSNorm
        rms_w_offset = layer * dim;
        xb = rmsnorm(x, rms_ffn_w, xb, dim, 0.00001, rms_w_offset);

        // FFN: SwiGLU
        var ffn_offset = layer * dim * hidden_dim;

        // Zero output buffer for hb
        i = 0i;
        while (i < hidden_dim) {
            hb[i] = 0.0;
            i = i + 1i;
        }

        // hb = w1 @ xb (xb: [dim x 1], w1: [hidden_dim x dim])
        hb = matmul(w1, xb, hb, 8192i, 2048i, 1i, ffn_offset, 0i, 0i);

        // hb_silu = silu(hb)
        hb_silu = silu(hb, hb_silu, hidden_dim);

        // Reuse hb for w3 output (we already saved silu result in hb_silu)
        // Zero hb for w3 output
        i = 0i;
        while (i < hidden_dim) {
            hb[i] = 0.0;
            i = i + 1i;
        }

        // hb = w3 @ xb (reusing hb as temp buffer)
        hb = matmul(w3, xb, hb, 8192i, 2048i, 1i, ffn_offset, 0i, 0i);

        // Element-wise multiply: hb = hb_silu * hb (w3 output)
        i = 0i;
        while (i < hidden_dim) {
            hb[i] = hb_silu[i] * hb[i];
            i = i + 1i;
        }

        // Down projection: xb = w2 @ hb (hb: [hidden_dim x 1], w2: [dim x hidden_dim])
        var w2_offset = layer * hidden_dim * dim;
        // Zero output buffer
        i = 0i;
        while (i < dim) {
            xb[i] = 0.0;
            i = i + 1i;
        }
        xb = matmul(w2, hb, xb, 2048i, 8192i, 1i, w2_offset, 0i, 0i);

        // Residual connection
        i = 0i;
        while (i < dim) {
            x[i] = x[i] + xb[i];
            i = i + 1i;
        }

        layer = layer + 1i;
    }

    // Final RMSNorm (no layer offset for final norm)
    xb = rmsnorm(x, rms_final_w, xb, dim, 0.00001, 0i);

    // Classifier: logits = wcls @ xb (xb: [dim x 1], wcls: [vocab_size x dim])
    // Zero output buffer
    i = 0i;
    while (i < vocab_size) {
        logits[i] = 0.0;
        i = i + 1i;
    }
    logits = matmul(wcls, xb, logits, 32000i, 2048i, 1i, 0i, 0i, 0i);

    return logits[0];
}
