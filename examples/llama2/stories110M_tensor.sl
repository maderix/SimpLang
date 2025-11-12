// Llama2 110M model inference using tensor operations
// Input: x (token embeddings), output: logits
fn stories110M_forward(
    // Token embeddings
    f32[] token_embedding_table,

    // Transformer weights (12 layers)
    f32[] rms_att_weight,  // [n_layers, dim]
    f32[] wq, f32[] wk, f32[] wv, f32[] wo,  // attention weights
    f32[] rms_ffn_weight,  // [n_layers, dim]
    f32[] w1, f32[] w2, f32[] w3,  // FFN weights

    // Final layer
    f32[] rms_final_weight,  // [dim]
    f32[] wcls,  // classifier weights [vocab_size, dim]

    // Activations and state (passed from host)
    f32[] x,        // input activation [dim]
    f32[] xb,       // intermediate buffer [dim]
    f32[] xb2,      // second buffer [dim]
    f32[] hb,       // hidden dimension buffer [hidden_dim]
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
    i64 dim,        // 768
    i64 hidden_dim, // 2048 for stories110M
    i64 n_layers,   // 12
    i64 n_heads,    // 12
    i64 n_kv_heads, // 12
    i64 vocab_size, // 32000
    i64 seq_len     // 1024
) -> f32 {
    // Derived hyperparameters
    var head_size = dim / n_heads;  // 64
    var kv_dim = (dim * n_kv_heads) / n_heads;  // 768 for stories110M

    // Get token embedding: x = token_embedding_table[token, :]
    var i = 0i;
    while (i < dim) {
        x[i] = token_embedding_table[token * dim + i];
        i = i + 1i;
    }

    // Forward pass through all layers
    var layer = 0i;
    while (layer < n_layers) {
        // Attention: RMS norm
        var rms_offset = layer * dim;
        xb = rmsnorm(x, rms_att_weight, xb, dim, 0.00001, rms_offset);

        // QKV projections using tensor operations
        var qkv_offset = layer * dim * dim;

        // Extract weight tensors for this layer
        f32<768, 768> wq_layer = tensor_from_array(wq, qkv_offset);
        f32<768, 768> wk_layer = tensor_from_array(wk, qkv_offset);
        f32<768, 768> wv_layer = tensor_from_array(wv, qkv_offset);

        // Convert xb to tensor for matmul
        f32<768, 1> xb_vec = tensor_from_array(xb, 0i);

        // Compute Q, K, V using tensor matmul
        f32<768, 1> q_tensor = tensor_matmul(wq_layer, xb_vec);
        f32<768, 1> k_tensor = tensor_matmul(wk_layer, xb_vec);
        f32<768, 1> v_tensor = tensor_matmul(wv_layer, xb_vec);

        // Copy tensor results back to arrays (for RoPE and attention)
        i = 0i;
        while (i < dim) {
            q[i] = q_tensor[i, 0i];
            k[i] = k_tensor[i, 0i];
            v[i] = v_tensor[i, 0i];
            i = i + 1i;
        }

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

        // Multi-head attention (keep as loops - tensor ops don't help here)
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
                score = score / 8.0;  // sqrt(head_size=64) = 8.0
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

        // Output projection using tensor operations
        var wo_offset = layer * dim * dim;
        f32<768, 768> wo_layer = tensor_from_array(wo, wo_offset);
        f32<768, 1> xb_vec2 = tensor_from_array(xb, 0i);  // Update xb_vec with attention output
        f32<768, 1> xb2_tensor = tensor_matmul(wo_layer, xb_vec2);

        // Copy back and add residual
        i = 0i;
        while (i < dim) {
            x[i] = x[i] + xb2_tensor[i, 0i];
            i = i + 1i;
        }

        // FFN: RMS norm
        xb = rmsnorm(x, rms_ffn_weight, xb, dim, 0.00001, rms_offset);

        // FFN layers using tensor operations
        var ffn_offset = layer * hidden_dim * dim;
        f32<2048, 768> w1_layer = tensor_from_array(w1, ffn_offset);
        f32<2048, 768> w3_layer = tensor_from_array(w3, ffn_offset);

        f32<768, 1> xb_vec3 = tensor_from_array(xb, 0i);

        // hb = w1 @ xb (with SiLU activation)
        f32<2048, 1> hb_tensor = tensor_matmul(w1_layer, xb_vec3);

        // xb2 = w3 @ xb
        f32<2048, 1> xb2_2048_tensor = tensor_matmul(w3_layer, xb_vec3);

        // Apply SiLU and elementwise multiply: hb = SiLU(hb) * xb2
        i = 0i;
        while (i < hidden_dim) {
            var h_val = hb_tensor[i, 0i];
            var silu = h_val / (1.0 + exp(0.0 - h_val));  // SiLU(x) = x / (1 + e^-x)
            hb[i] = silu * xb2_2048_tensor[i, 0i];
            i = i + 1i;
        }

        // Final FFN projection: xb = w2 @ hb
        var w2_offset = layer * dim * hidden_dim;
        f32<768, 2048> w2_layer = tensor_from_array(w2, w2_offset);
        f32<2048, 1> hb_vec = tensor_from_array(hb, 0i);

        f32<768, 1> xb_out_tensor = tensor_matmul(w2_layer, hb_vec);

        // Residual connection
        i = 0i;
        while (i < dim) {
            x[i] = x[i] + xb_out_tensor[i, 0i];
            i = i + 1i;
        }

        layer = layer + 1i;
    }

    // Final RMS norm
    xb = rmsnorm(x, rms_final_weight, xb, dim, 0.00001, 0i);

    // Classifier using tensor operations
    f32<32000, 768> wcls_tensor = tensor_from_array(wcls, 0i);
    f32<768, 1> xb_vec4 = tensor_from_array(xb, 0i);
    f32<32000, 1> logits_tensor = tensor_matmul(wcls_tensor, xb_vec4);

    // Copy logits to output array
    i = 0i;
    while (i < vocab_size) {
        logits[i] = logits_tensor[i, 0i];
        i = i + 1i;
    }

    return logits[0];
}
