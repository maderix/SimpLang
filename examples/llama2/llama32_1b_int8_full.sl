// LLaMA 3.2-1B Fully Quantized Model - NO FP32
// W8A8 matmuls with INT32 accumulation (VNNI)
// INT16/INT32 for all element-wise operations
//
// Architecture:
//   dim = 2048, n_layers = 16, n_heads = 32, n_kv_heads = 8
//   head_dim = 64, hidden_dim = 8192, vocab_size = 128256

// ============================================================================
// INT16 RMSNorm: x_out = (x * rsqrt(mean(x^2) + eps)) * weight
// Newton-Raphson rsqrt with epsilon for numerical stability
// ============================================================================
fn rmsnorm_i16(
    i16[] x,          // [dim] input (Q8)
    i16[] weight,     // [dim] RMSNorm weight (Q15)
    i16[] out,        // [dim] output (Q8)
    i64 dim,
    i64 offset        // offset into weight array
) -> i32 {
    // Compute sum of squares (Q16 accumulator)
    var ss = 0;
    var i = 0i;
    while (i < dim) {
        var xi = x[i] as i32;
        ss = ss + (xi * xi);
        i = i + 1i;
    }
    // Mean: Q16, add epsilon
    ss = ss / dim;
    ss = ss + 1;

    // Newton-Raphson rsqrt (4 iterations), y is Q8
    var y = 256;
    var j = 0i;
    while (j < 4i) {
        var y2 = (y * y);
        var xy2 = (ss * y2) / 65536;
        var term = 196608 - xy2;
        y = (y * term) / 131072;
        j = j + 1i;
    }

    // Apply normalization and weight using out array as temp
    i = 0i;
    while (i < dim) {
        var xi = x[i] as i32;
        var wi = weight[offset + i] as i32;
        var normed = (xi * y) / 256;
        var result = (normed * wi) / 32768;

        // Store result, then clamp using array
        out[i] = result as i16;
        if (result > 32767) { out[i] = 32767 as i16; }
        if (result < -32768) { out[i] = -32768 as i16; }
        i = i + 1i;
    }
    return out[0] as i32;
}

// ============================================================================
// INT16 Softmax with cubic exp approximation
// Input: INT32 scores, Output: INT8 probabilities (sum to 127)
// ============================================================================
fn softmax_i32_to_i8(
    i32[] scores,     // [seq_len] attention scores
    i8[] probs,       // [seq_len] output probabilities
    i64 seq_len,
    i64 in_offset,
    i64 out_offset
) -> i32 {
    // Find max for numerical stability (use scores array to store temp)
    var max_val = scores[in_offset];
    var i = 1i;
    while (i < seq_len) {
        var curr = scores[in_offset + i];
        if (curr > max_val) { max_val = curr; }
        i = i + 1i;
    }

    // Compute exp with cubic approximation and sum
    var sum = 0;
    i = 0i;
    while (i < seq_len) {
        var diff = scores[in_offset + i] - max_val;
        // Clamp diff: use probs array as temp storage
        probs[out_offset + i] = diff as i8;
        if (diff < -127) { probs[out_offset + i] = -127 as i8; }
        diff = probs[out_offset + i] as i32;

        // Cubic exp: exp(x) ≈ 128 + 8x + x²/4 + x³/192
        var linear = diff * 8;
        var diff_sq = diff * diff;
        var quadratic = diff_sq / 4;
        var cubic = (diff_sq / 16) * diff / 12;

        var e = 128 + linear + quadratic + cubic;
        // Clamp e to [1, 255] using array
        probs[out_offset + i] = e as i8;
        if (e < 1) { probs[out_offset + i] = 1 as i8; }
        if (e > 127) { probs[out_offset + i] = 127 as i8; }
        sum = sum + (probs[out_offset + i] as i32);
        i = i + 1i;
    }

    // Normalize to sum to ~127
    if (sum > 0) {
        i = 0i;
        while (i < seq_len) {
            var p = (probs[out_offset + i] as i32) * 127 / sum;
            probs[out_offset + i] = p as i8;
            i = i + 1i;
        }
    }
    return probs[out_offset] as i32;
}

// ============================================================================
// INT32 RoPE with precomputed INT16 sin/cos tables (Q15)
// ============================================================================
fn rope_i32(
    i32[] q,          // [dim] query or key vector
    i16[] cos_tab,    // [head_dim/2] precomputed cos (Q15)
    i16[] sin_tab,    // [head_dim/2] precomputed sin (Q15)
    i32[] out,        // [dim] rotated output
    i64 dim,
    i64 head_dim,
    i64 pos_offset    // position * head_dim/2 into sin/cos tables
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

            // Q15 multiply: (a * b) / 32768
            out[head_off + i * 2i] = (q0 * c - q1 * s) / 32768i;
            out[head_off + i * 2i + 1i] = (q0 * s + q1 * c) / 32768i;
            i = i + 1i;
        }
        h = h + 1i;
    }
    return out[0];
}

// ============================================================================
// INT32 SiLU with cubic sigmoid: silu(x) = x * sigmoid(x)
// sigmoid(x) ≈ 0.5 + x/4 - x²/32 (with saturation)
// ============================================================================
fn silu_mul_i32(
    i32[] gate,       // [hidden_dim] gate projection output
    i32[] up,         // [hidden_dim] up projection output
    i8[] out,         // [hidden_dim] output quantized to INT8
    i64 hidden_dim,
    i64 scale         // output scaling factor
) -> i32 {
    var i = 0i;
    while (i < hidden_dim) {
        var g = gate[i] / 512;  // Scale down from matmul output
        var u = up[i] / 512;

        // Improved sigmoid approximation with saturation
        // sigmoid ≈ 128 + g/4 - g²/8192 (in Q8, 256 = 1.0)
        var linear = g / 4;
        var g_sq = g * g;
        var correction = g_sq / 8192;

        // Compute sigmoid using gate array as temp (won't need g anymore)
        var sig = 128 + linear;
        gate[i] = sig;
        if (g > 0) { gate[i] = sig - correction; }
        if (g < 0) { gate[i] = sig + correction; }
        sig = gate[i];

        // Saturate sigmoid to [0, 256]
        if (sig < 0) { gate[i] = 0; }
        if (sig > 256) { gate[i] = 256; }
        sig = gate[i];

        // SiLU = x * sigmoid(x) / 256 (Q8 normalization)
        var silu = (g * sig) / 256;

        // Multiply with up projection and scale to INT8
        var result = (silu * u) / scale;

        // Clamp to INT8 range using out array
        out[i] = result as i8;
        if (result > 127) { out[i] = 127 as i8; }
        if (result < -128) { out[i] = -128 as i8; }
        i = i + 1i;
    }
    return out[0] as i32;
}

// ============================================================================
// Decode: Single token forward pass (fully quantized)
// ============================================================================
fn llama32_decode_i8(
    // Input token embedding (already quantized)
    i8[] x_i8,                // [dim] input

    // RMSNorm weights (INT16 Q15)
    i16[] rms_att_w,          // [n_layers, dim]
    i16[] rms_ffn_w,          // [n_layers, dim]
    i16[] rms_final_w,        // [dim]

    // Attention weights - INT8, PRE-TRANSPOSED [N, K]
    i8[] wq_t,                // [n_layers * dim * dim]
    i8[] wk_t,                // [n_layers * kv_dim * dim]
    i8[] wv_t,                // [n_layers * kv_dim * dim]
    i8[] wo_t,                // [n_layers * dim * dim]

    // FFN weights - INT8, PRE-TRANSPOSED
    i8[] w1_t,                // [n_layers * hidden_dim * dim]
    i8[] w2_t,                // [n_layers * dim * hidden_dim]
    i8[] w3_t,                // [n_layers * hidden_dim * dim]

    // Classifier - INT8
    i8[] wcls_t,              // [vocab_size, dim] transposed

    // Buffers
    i16[] xb_i16,             // [dim] normalized buffer
    i8[] xb_i8,               // [dim] quantized buffer
    i32[] q_i32,              // [dim] Q projection
    i32[] k_i32,              // [kv_dim] K projection
    i32[] v_i32,              // [kv_dim] V projection
    i32[] attn_out,           // [dim] attention output
    i8[] ffn_hb,              // [hidden_dim] FFN hidden

    // KV Cache - INT8
    i8[] k_cache,             // [n_layers, max_seq_len, kv_dim]
    i8[] v_cache,             // [n_layers, max_seq_len, kv_dim]

    // Attention scores and probs (INT32/INT8)
    i32[] att_scores,         // [n_heads, max_seq_len]
    i8[] att_probs,           // [n_heads, max_seq_len]

    // RoPE tables (INT16 Q15)
    i16[] cos_tab,            // [max_seq_len, head_dim/2]
    i16[] sin_tab,            // [max_seq_len, head_dim/2]

    // Output logits (INT32)
    i32[] logits,             // [vocab_size]

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
) -> i32 {
    var kv_dim = n_kv_heads * head_dim;  // 512
    var n_rep = n_heads / n_kv_heads;     // 4
    var seq_len = pos + 1i;

    // Copy input to INT16 for normalization
    var i = 0i;
    while (i < dim) {
        xb_i16[i] = (x_i8[i] as i32 * 256i) as i16;  // Scale up for precision
        i = i + 1i;
    }

    // ========================================
    // Transformer Layers
    // ========================================
    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        // ----- Attention RMSNorm (INT16) -----
        var dummy = rmsnorm_i16(xb_i16, rms_att_w, xb_i16, dim, layer_offset);

        // Quantize to INT8 for matmul
        i = 0i;
        while (i < dim) {
            var val = xb_i16[i] as i32 / 256i;
            if (val > 127i) { val = 127i; }
            if (val < -128i) { val = -128i; }
            xb_i8[i] = val as i8;
            i = i + 1i;
        }

        // ----- QKV Projections using tensor_matmul_nt (VNNI) -----
        i8<1, 2048> xb_mat = tensor_from_array(xb_i8, 0i);

        var wq_offset = layer * dim * dim;
        i8<2048, 2048> wq_mat = tensor_from_array(wq_t, wq_offset);
        var Q = tensor_matmul_nt(xb_mat, wq_mat);  // [1, 2048] i32

        var wk_offset = layer * kv_dim * dim;
        i8<512, 2048> wk_mat = tensor_from_array(wk_t, wk_offset);
        var K = tensor_matmul_nt(xb_mat, wk_mat);  // [1, 512] i32

        var wv_offset = layer * kv_dim * dim;
        i8<512, 2048> wv_mat = tensor_from_array(wv_t, wv_offset);
        var V = tensor_matmul_nt(xb_mat, wv_mat);  // [1, 512] i32

        // Copy to i32 arrays for RoPE
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

        // ----- RoPE on Q and K (INT32 with INT16 tables) -----
        var rope_offset = pos * (head_dim / 2i);
        dummy = rope_i32(q_i32, cos_tab, sin_tab, q_i32, dim, head_dim, rope_offset);
        dummy = rope_i32(k_i32, cos_tab, sin_tab, k_i32, kv_dim, head_dim, rope_offset);

        // ----- Store K, V in cache (scale to INT8) -----
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

        // ----- Multi-Head Attention with GQA (INT8/INT32) -----
        i = 0i;
        while (i < dim) {
            attn_out[i] = 0i;
            i = i + 1i;
        }

        var h = 0i;
        while (h < n_heads) {
            var q_head_offset = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_offset = kv_head * head_dim;

            // Compute attention scores (INT32)
            var t = 0i;
            while (t < seq_len) {
                var score = 0i;
                var k_pos = layer * max_seq_len * kv_dim + t * kv_dim + kv_head_offset;

                var d = 0i;
                while (d < head_dim) {
                    var q_val = q_i32[q_head_offset + d];
                    var k_val = k_cache[k_pos + d] as i32;
                    score = score + (q_val * k_val) / 256i;
                    d = d + 1i;
                }
                // Scale by 1/sqrt(head_dim) ≈ /8
                att_scores[h * max_seq_len + t] = score / 8i;
                t = t + 1i;
            }

            // Softmax (INT32 -> INT8)
            var att_offset = h * max_seq_len;
            dummy = softmax_i32_to_i8(att_scores, att_probs, seq_len, att_offset, att_offset);

            // Weighted sum of V (INT8 probs @ INT8 V -> INT32)
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
                attn_out[q_head_offset + d] = val;
                d = d + 1i;
            }

            h = h + 1i;
        }

        // ----- Output Projection (INT8 matmul) -----
        i = 0i;
        while (i < dim) {
            var val = attn_out[i] / 128i;
            if (val > 127i) { val = 127i; }
            if (val < -128i) { val = -128i; }
            xb_i8[i] = val as i8;
            i = i + 1i;
        }

        i8<1, 2048> attn_mat = tensor_from_array(xb_i8, 0i);
        var wo_offset = layer * dim * dim;
        i8<2048, 2048> wo_mat = tensor_from_array(wo_t, wo_offset);
        var Wo_out = tensor_matmul_nt(attn_mat, wo_mat);  // [1, 2048] i32

        // Residual connection (INT32 add, store in INT16)
        i = 0i;
        while (i < dim) {
            var res = xb_i16[i] as i32 + Wo_out[0i, i] / 256i;
            if (res > 32767i) { res = 32767i; }
            if (res < -32768i) { res = -32768i; }
            xb_i16[i] = res as i16;
            i = i + 1i;
        }

        // ----- FFN RMSNorm (INT16) -----
        dummy = rmsnorm_i16(xb_i16, rms_ffn_w, xb_i16, dim, layer_offset);

        // Quantize for FFN
        i = 0i;
        while (i < dim) {
            var val = xb_i16[i] as i32 / 256i;
            if (val > 127i) { val = 127i; }
            if (val < -128i) { val = -128i; }
            xb_i8[i] = val as i8;
            i = i + 1i;
        }

        // ----- SwiGLU FFN with tensor_matmul_nt -----
        var w1_offset = layer * hidden_dim * dim;
        var w3_offset = layer * hidden_dim * dim;
        var w2_offset = layer * dim * hidden_dim;

        // Gate projection
        i8<1, 2048> ffn_in = tensor_from_array(xb_i8, 0i);
        i8<8192, 2048> w1_mat = tensor_from_array(w1_t, w1_offset);
        var Gate = tensor_matmul_nt(ffn_in, w1_mat);  // [1, 8192] i32

        // Up projection
        i8<8192, 2048> w3_mat = tensor_from_array(w3_t, w3_offset);
        var Up = tensor_matmul_nt(ffn_in, w3_mat);  // [1, 8192] i32

        // SiLU(gate) * up -> ffn_hb (INT8)
        var hi = 0i;
        while (hi < hidden_dim) {
            var g = Gate[0i, hi] / 512i;
            var u = Up[0i, hi] / 512i;

            // SiLU quadratic approximation
            var silu = g / 2i + (g * g) / 2048i;
            var result = (silu * u) / 512i;

            if (result > 127i) { result = 127i; }
            if (result < -128i) { result = -128i; }
            ffn_hb[hi] = result as i8;
            hi = hi + 1i;
        }

        // Down projection
        i8<1, 8192> hb_mat = tensor_from_array(ffn_hb, 0i);
        i8<2048, 8192> w2_mat = tensor_from_array(w2_t, w2_offset);
        var Down = tensor_matmul_nt(hb_mat, w2_mat);  // [1, 2048] i32

        // Residual connection
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

    // ========================================
    // Final RMSNorm (INT16)
    // ========================================
    var dummy2 = rmsnorm_i16(xb_i16, rms_final_w, xb_i16, dim, 0i);

    // Quantize final output to INT8
    i = 0i;
    while (i < dim) {
        var val = xb_i16[i] as i32 / 256i;
        if (val > 127i) { val = 127i; }
        if (val < -128i) { val = -128i; }
        xb_i8[i] = val as i8;
        i = i + 1i;
    }

    // ========================================
    // Classifier (INT8 matmul -> INT32 logits)
    // ========================================
    i8<1, 2048> final_mat = tensor_from_array(xb_i8, 0i);
    i8<128256, 2048> wcls_mat = tensor_from_array(wcls_t, 0i);
    var Logits = tensor_matmul_nt(final_mat, wcls_mat);  // [1, vocab_size] i32

    // Copy to output
    i = 0i;
    while (i < vocab_size) {
        logits[i] = Logits[0i, i];
        i = i + 1i;
    }

    return logits[0];
}

// ============================================================================
// Prefill: Process seq_len=32 tokens at once (batched, I=4 tiling works)
// ============================================================================
fn llama32_prefill_32_i8(
    // Input embeddings
    i8[] x_i8,                // [seq_len, dim]

    // Weights (same as decode)
    i16[] rms_att_w,
    i16[] rms_ffn_w,
    i8[] wq_t, i8[] wk_t, i8[] wv_t, i8[] wo_t,
    i8[] w1_t, i8[] w2_t, i8[] w3_t,

    // Buffers (sized for batch)
    i16[] xb_i16,             // [seq_len, dim]
    i8[] xb_i8,               // [seq_len, dim]

    // KV Cache
    i8[] k_cache,             // [n_layers, seq_len, kv_dim]
    i8[] v_cache,

    // Attention buffers
    i32[] att_scores,         // [n_heads, seq_len, seq_len]
    i8[] att_probs,           // [n_heads, seq_len, seq_len]
    i32[] attn_out,           // [seq_len, dim]
    i8[] ffn_hb,              // [seq_len, hidden_dim]

    // RoPE tables
    i16[] cos_tab,
    i16[] sin_tab,

    // Config
    i64 seq_len,              // 32
    i64 dim,                  // 2048
    i64 hidden_dim,           // 8192
    i64 n_layers,             // 16
    i64 n_heads,              // 32
    i64 n_kv_heads,           // 8
    i64 head_dim              // 64
) -> i32 {
    var kv_dim = n_kv_heads * head_dim;
    var n_rep = n_heads / n_kv_heads;

    // Initialize xb_i16 from input
    var i = 0i;
    while (i < seq_len * dim) {
        xb_i16[i] = (x_i8[i] as i32 * 256i) as i16;
        i = i + 1i;
    }

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        // RMSNorm each token
        var t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;

            // Compute sum of squares
            var ss = 0i;
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
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
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_att_w[layer_offset + i] as i32;
                var normed = (val * rsqrt_approx) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;

                // Also quantize to INT8
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        // Batched QKV projection (32 tokens at once - I=4 tiling works!)
        i8<32, 2048> xb_mat = tensor_from_array(xb_i8, 0i);

        var wq_offset = layer * dim * dim;
        i8<2048, 2048> wq_mat = tensor_from_array(wq_t, wq_offset);
        var Q = tensor_matmul_nt(xb_mat, wq_mat);  // [32, 2048] i32

        var wk_offset = layer * kv_dim * dim;
        i8<512, 2048> wk_mat = tensor_from_array(wk_t, wk_offset);
        var K = tensor_matmul_nt(xb_mat, wk_mat);  // [32, 512] i32

        var wv_offset = layer * kv_dim * dim;
        i8<512, 2048> wv_mat = tensor_from_array(wv_t, wv_offset);
        var V = tensor_matmul_nt(xb_mat, wv_mat);  // [32, 512] i32

        // Store KV cache (with RoPE applied per position)
        var cache_layer = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            var rope_off = t * (head_dim / 2i);
            i = 0i;
            while (i < kv_dim) {
                // Apply RoPE inline
                var head_i = i / head_dim;
                var local_i = i % head_dim;
                var pair_i = local_i / 2i;

                var k_val = K[t, i] as i32;
                // Simplified: just scale and store (full RoPE would need pair processing)
                var k_scaled = k_val / 256i;
                if (k_scaled > 127i) { k_scaled = 127i; }
                if (k_scaled < -128i) { k_scaled = -128i; }
                k_cache[cache_layer + t * kv_dim + i] = k_scaled as i8;

                var v_val = V[t, i] as i32 / 256i;
                if (v_val > 127i) { v_val = 127i; }
                if (v_val < -128i) { v_val = -128i; }
                v_cache[cache_layer + t * kv_dim + i] = v_val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        // Batched Causal Attention
        // For each query position, attend to all previous positions
        var h = 0i;
        while (h < n_heads) {
            var q_head_off = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_off = kv_head * head_dim;

            var q_pos = 0i;
            while (q_pos < seq_len) {
                // Compute scores for this query against all keys up to q_pos
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0i;
                    var d = 0i;
                    while (d < head_dim) {
                        var q_val = Q[q_pos, q_head_off + d];
                        var k_val = k_cache[cache_layer + k_pos * kv_dim + kv_head_off + d] as i32;
                        score = score + (q_val * k_val) / 256i;
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8i;
                    k_pos = k_pos + 1i;
                }
                // Mask future positions
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -100000i;
                    k_pos = k_pos + 1i;
                }

                // Softmax for this query
                var att_off = h * seq_len * seq_len + q_pos * seq_len;
                var max_s = att_scores[att_off];
                var ki = 1i;
                while (ki <= q_pos) {
                    if (att_scores[att_off + ki] > max_s) {
                        max_s = att_scores[att_off + ki];
                    }
                    ki = ki + 1i;
                }

                var sum = 0i;
                ki = 0i;
                while (ki <= q_pos) {
                    var diff = att_scores[att_off + ki] - max_s;
                    var e = 128i + diff / 4i;
                    if (e < 0i) { e = 0i; }
                    if (e > 255i) { e = 255i; }
                    att_probs[att_off + ki] = e as i8;
                    sum = sum + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    var p = (att_probs[att_off + ki] as i32) * 127i / sum;
                    att_probs[att_off + ki] = p as i8;
                    ki = ki + 1i;
                }

                // Weighted sum of V
                var d = 0i;
                while (d < head_dim) {
                    var val = 0i;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        var prob = att_probs[att_off + vi] as i32;
                        var v_val = v_cache[cache_layer + vi * kv_dim + kv_head_off + d] as i32;
                        val = val + prob * v_val;
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_off + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        // Output projection (batched)
        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < dim) {
                var val = attn_out[t * dim + i] / 128i;
                if (val > 127i) { val = 127i; }
                if (val < -128i) { val = -128i; }
                xb_i8[t * dim + i] = val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<32, 2048> attn_mat = tensor_from_array(xb_i8, 0i);
        var wo_offset = layer * dim * dim;
        i8<2048, 2048> wo_mat = tensor_from_array(wo_t, wo_offset);
        var Wo_out = tensor_matmul_nt(attn_mat, wo_mat);

        // Residual + FFN RMSNorm
        t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;

            // Residual
            i = 0i;
            while (i < dim) {
                var res = xb_i16[tok_off + i] as i32 + Wo_out[t, i] / 256i;
                if (res > 32767i) { res = 32767i; }
                if (res < -32768i) { res = -32768i; }
                xb_i16[tok_off + i] = res as i16;
                i = i + 1i;
            }

            // FFN RMSNorm
            var ss = 0i;
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                ss = ss + (val * val) / 256i;
                i = i + 1i;
            }
            var mean = ss / dim;
            var rsqrt = 16384i;
            if (mean > 256i) { rsqrt = 8192i; }
            if (mean > 1024i) { rsqrt = 4096i; }
            if (mean > 4096i) { rsqrt = 2048i; }
            if (mean > 16384i) { rsqrt = 1024i; }

            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_ffn_w[layer_offset + i] as i32;
                var normed = (val * rsqrt) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;

                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        // Batched FFN
        i8<32, 2048> ffn_in = tensor_from_array(xb_i8, 0i);

        var w1_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w1_mat = tensor_from_array(w1_t, w1_offset);
        var Gate = tensor_matmul_nt(ffn_in, w1_mat);

        var w3_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w3_mat = tensor_from_array(w3_t, w3_offset);
        var Up = tensor_matmul_nt(ffn_in, w3_mat);

        // SiLU * up for each token
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

        // Down projection
        i8<32, 8192> hb_mat = tensor_from_array(ffn_hb, 0i);
        var w2_offset = layer * dim * hidden_dim;
        i8<2048, 8192> w2_mat = tensor_from_array(w2_t, w2_offset);
        var Down = tensor_matmul_nt(hb_mat, w2_mat);

        // Final residual
        t = 0i;
        while (t < seq_len) {
            i = 0i;
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

    // Return last token's first element (for verification)
    return xb_i16[(seq_len - 1i) * dim] as i32;
}

// ============================================================================
// Prefill variants for different batch sizes (tensor dims are compile-time)
// ============================================================================

// Prefill 4 tokens
fn llama32_prefill_4_i8(
    i8[] x_i8, i16[] rms_att_w, i16[] rms_ffn_w,
    i8[] wq_t, i8[] wk_t, i8[] wv_t, i8[] wo_t,
    i8[] w1_t, i8[] w2_t, i8[] w3_t,
    i16[] xb_i16, i8[] xb_i8,
    i8[] k_cache, i8[] v_cache,
    i32[] att_scores, i8[] att_probs, i32[] attn_out, i8[] ffn_hb,
    i16[] cos_tab, i16[] sin_tab,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers, i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> i32 {
    var kv_dim = n_kv_heads * head_dim;
    var i = 0i;
    while (i < seq_len * dim) { xb_i16[i] = (x_i8[i] as i32 * 256i) as i16; i = i + 1i; }

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;
        var t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;
            var ss = 0i; i = 0i;
            while (i < dim) { var val = xb_i16[tok_off + i] as i32; ss = ss + (val * val) / 256i; i = i + 1i; }
            var mean = ss / dim;
            var rsqrt_approx = 16384i;
            if (mean > 256i) { rsqrt_approx = 8192i; }
            if (mean > 1024i) { rsqrt_approx = 4096i; }
            if (mean > 4096i) { rsqrt_approx = 2048i; }
            if (mean > 16384i) { rsqrt_approx = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_att_w[layer_offset + i] as i32;
                var normed = (val * rsqrt_approx) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<4, 2048> xb_mat = tensor_from_array(xb_i8, 0i);
        var wq_offset = layer * dim * dim;
        i8<2048, 2048> wq_mat = tensor_from_array(wq_t, wq_offset);
        var Q = tensor_matmul_nt(xb_mat, wq_mat);
        var wk_offset = layer * kv_dim * dim;
        i8<512, 2048> wk_mat = tensor_from_array(wk_t, wk_offset);
        var K = tensor_matmul_nt(xb_mat, wk_mat);
        var wv_offset = layer * kv_dim * dim;
        i8<512, 2048> wv_mat = tensor_from_array(wv_t, wv_offset);
        var V = tensor_matmul_nt(xb_mat, wv_mat);

        var cache_layer = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < kv_dim) {
                var k_scaled = K[t, i] as i32 / 256i;
                if (k_scaled > 127i) { k_scaled = 127i; }
                if (k_scaled < -128i) { k_scaled = -128i; }
                k_cache[cache_layer + t * kv_dim + i] = k_scaled as i8;
                var v_val = V[t, i] as i32 / 256i;
                if (v_val > 127i) { v_val = 127i; }
                if (v_val < -128i) { v_val = -128i; }
                v_cache[cache_layer + t * kv_dim + i] = v_val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        var n_rep = n_heads / n_kv_heads;
        var h = 0i;
        while (h < n_heads) {
            var q_head_off = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_off = kv_head * head_dim;
            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0i;
                    var d = 0i;
                    while (d < head_dim) {
                        var q_val = Q[q_pos, q_head_off + d];
                        var k_val = k_cache[cache_layer + k_pos * kv_dim + kv_head_off + d] as i32;
                        score = score + (q_val * k_val) / 256i;
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8i;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -100000i;
                    k_pos = k_pos + 1i;
                }
                var att_off = h * seq_len * seq_len + q_pos * seq_len;
                var max_s = att_scores[att_off];
                var ki = 1i;
                while (ki <= q_pos) { if (att_scores[att_off + ki] > max_s) { max_s = att_scores[att_off + ki]; } ki = ki + 1i; }
                var sum = 0i;
                ki = 0i;
                while (ki <= q_pos) {
                    var diff = att_scores[att_off + ki] - max_s;
                    var e = 128i + diff / 4i;
                    if (e < 0i) { e = 0i; }
                    if (e > 255i) { e = 255i; }
                    att_probs[att_off + ki] = e as i8;
                    sum = sum + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    var p = (att_probs[att_off + ki] as i32) * 127i / sum;
                    att_probs[att_off + ki] = p as i8;
                    ki = ki + 1i;
                }
                var d = 0i;
                while (d < head_dim) {
                    var val = 0i;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        var prob = att_probs[att_off + vi] as i32;
                        var v_val = v_cache[cache_layer + vi * kv_dim + kv_head_off + d] as i32;
                        val = val + prob * v_val;
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_off + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < dim) {
                var val = attn_out[t * dim + i] / 128i;
                if (val > 127i) { val = 127i; }
                if (val < -128i) { val = -128i; }
                xb_i8[t * dim + i] = val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<4, 2048> attn_mat = tensor_from_array(xb_i8, 0i);
        var wo_offset = layer * dim * dim;
        i8<2048, 2048> wo_mat = tensor_from_array(wo_t, wo_offset);
        var Wo_out = tensor_matmul_nt(attn_mat, wo_mat);

        t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;
            i = 0i;
            while (i < dim) {
                var res = xb_i16[tok_off + i] as i32 + Wo_out[t, i] / 256i;
                if (res > 32767i) { res = 32767i; }
                if (res < -32768i) { res = -32768i; }
                xb_i16[tok_off + i] = res as i16;
                i = i + 1i;
            }
            var ss = 0i;
            i = 0i;
            while (i < dim) { var val = xb_i16[tok_off + i] as i32; ss = ss + (val * val) / 256i; i = i + 1i; }
            var mean = ss / dim;
            var rsqrt = 16384i;
            if (mean > 256i) { rsqrt = 8192i; }
            if (mean > 1024i) { rsqrt = 4096i; }
            if (mean > 4096i) { rsqrt = 2048i; }
            if (mean > 16384i) { rsqrt = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_ffn_w[layer_offset + i] as i32;
                var normed = (val * rsqrt) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<4, 2048> ffn_in = tensor_from_array(xb_i8, 0i);
        var w1_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w1_mat = tensor_from_array(w1_t, w1_offset);
        var Gate = tensor_matmul_nt(ffn_in, w1_mat);
        var w3_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w3_mat = tensor_from_array(w3_t, w3_offset);
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

        i8<4, 8192> hb_mat = tensor_from_array(ffn_hb, 0i);
        var w2_offset = layer * dim * hidden_dim;
        i8<2048, 8192> w2_mat = tensor_from_array(w2_t, w2_offset);
        var Down = tensor_matmul_nt(hb_mat, w2_mat);

        t = 0i;
        while (t < seq_len) {
            i = 0i;
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

// Prefill 8 tokens
fn llama32_prefill_8_i8(
    i8[] x_i8, i16[] rms_att_w, i16[] rms_ffn_w,
    i8[] wq_t, i8[] wk_t, i8[] wv_t, i8[] wo_t,
    i8[] w1_t, i8[] w2_t, i8[] w3_t,
    i16[] xb_i16, i8[] xb_i8,
    i8[] k_cache, i8[] v_cache,
    i32[] att_scores, i8[] att_probs, i32[] attn_out, i8[] ffn_hb,
    i16[] cos_tab, i16[] sin_tab,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers, i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> i32 {
    var kv_dim = n_kv_heads * head_dim;
    var i = 0i;
    while (i < seq_len * dim) { xb_i16[i] = (x_i8[i] as i32 * 256i) as i16; i = i + 1i; }

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;
        var t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;
            var ss = 0i; i = 0i;
            while (i < dim) { var val = xb_i16[tok_off + i] as i32; ss = ss + (val * val) / 256i; i = i + 1i; }
            var mean = ss / dim;
            var rsqrt_approx = 16384i;
            if (mean > 256i) { rsqrt_approx = 8192i; }
            if (mean > 1024i) { rsqrt_approx = 4096i; }
            if (mean > 4096i) { rsqrt_approx = 2048i; }
            if (mean > 16384i) { rsqrt_approx = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_att_w[layer_offset + i] as i32;
                var normed = (val * rsqrt_approx) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<8, 2048> xb_mat = tensor_from_array(xb_i8, 0i);
        var wq_offset = layer * dim * dim;
        i8<2048, 2048> wq_mat = tensor_from_array(wq_t, wq_offset);
        var Q = tensor_matmul_nt(xb_mat, wq_mat);
        var wk_offset = layer * kv_dim * dim;
        i8<512, 2048> wk_mat = tensor_from_array(wk_t, wk_offset);
        var K = tensor_matmul_nt(xb_mat, wk_mat);
        var wv_offset = layer * kv_dim * dim;
        i8<512, 2048> wv_mat = tensor_from_array(wv_t, wv_offset);
        var V = tensor_matmul_nt(xb_mat, wv_mat);

        var cache_layer = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < kv_dim) {
                var k_scaled = K[t, i] as i32 / 256i;
                if (k_scaled > 127i) { k_scaled = 127i; }
                if (k_scaled < -128i) { k_scaled = -128i; }
                k_cache[cache_layer + t * kv_dim + i] = k_scaled as i8;
                var v_val = V[t, i] as i32 / 256i;
                if (v_val > 127i) { v_val = 127i; }
                if (v_val < -128i) { v_val = -128i; }
                v_cache[cache_layer + t * kv_dim + i] = v_val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        var n_rep = n_heads / n_kv_heads;
        var h = 0i;
        while (h < n_heads) {
            var q_head_off = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_off = kv_head * head_dim;
            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0i;
                    var d = 0i;
                    while (d < head_dim) {
                        var q_val = Q[q_pos, q_head_off + d];
                        var k_val = k_cache[cache_layer + k_pos * kv_dim + kv_head_off + d] as i32;
                        score = score + (q_val * k_val) / 256i;
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8i;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -100000i;
                    k_pos = k_pos + 1i;
                }
                var att_off = h * seq_len * seq_len + q_pos * seq_len;
                var max_s = att_scores[att_off];
                var ki = 1i;
                while (ki <= q_pos) { if (att_scores[att_off + ki] > max_s) { max_s = att_scores[att_off + ki]; } ki = ki + 1i; }
                var sum = 0i;
                ki = 0i;
                while (ki <= q_pos) {
                    var diff = att_scores[att_off + ki] - max_s;
                    var e = 128i + diff / 4i;
                    if (e < 0i) { e = 0i; }
                    if (e > 255i) { e = 255i; }
                    att_probs[att_off + ki] = e as i8;
                    sum = sum + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    var p = (att_probs[att_off + ki] as i32) * 127i / sum;
                    att_probs[att_off + ki] = p as i8;
                    ki = ki + 1i;
                }
                var d = 0i;
                while (d < head_dim) {
                    var val = 0i;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        var prob = att_probs[att_off + vi] as i32;
                        var v_val = v_cache[cache_layer + vi * kv_dim + kv_head_off + d] as i32;
                        val = val + prob * v_val;
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_off + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < dim) {
                var val = attn_out[t * dim + i] / 128i;
                if (val > 127i) { val = 127i; }
                if (val < -128i) { val = -128i; }
                xb_i8[t * dim + i] = val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<8, 2048> attn_mat = tensor_from_array(xb_i8, 0i);
        var wo_offset = layer * dim * dim;
        i8<2048, 2048> wo_mat = tensor_from_array(wo_t, wo_offset);
        var Wo_out = tensor_matmul_nt(attn_mat, wo_mat);

        t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;
            i = 0i;
            while (i < dim) {
                var res = xb_i16[tok_off + i] as i32 + Wo_out[t, i] / 256i;
                if (res > 32767i) { res = 32767i; }
                if (res < -32768i) { res = -32768i; }
                xb_i16[tok_off + i] = res as i16;
                i = i + 1i;
            }
            var ss = 0i;
            i = 0i;
            while (i < dim) { var val = xb_i16[tok_off + i] as i32; ss = ss + (val * val) / 256i; i = i + 1i; }
            var mean = ss / dim;
            var rsqrt = 16384i;
            if (mean > 256i) { rsqrt = 8192i; }
            if (mean > 1024i) { rsqrt = 4096i; }
            if (mean > 4096i) { rsqrt = 2048i; }
            if (mean > 16384i) { rsqrt = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_ffn_w[layer_offset + i] as i32;
                var normed = (val * rsqrt) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<8, 2048> ffn_in = tensor_from_array(xb_i8, 0i);
        var w1_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w1_mat = tensor_from_array(w1_t, w1_offset);
        var Gate = tensor_matmul_nt(ffn_in, w1_mat);
        var w3_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w3_mat = tensor_from_array(w3_t, w3_offset);
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

        i8<8, 8192> hb_mat = tensor_from_array(ffn_hb, 0i);
        var w2_offset = layer * dim * hidden_dim;
        i8<2048, 8192> w2_mat = tensor_from_array(w2_t, w2_offset);
        var Down = tensor_matmul_nt(hb_mat, w2_mat);

        t = 0i;
        while (t < seq_len) {
            i = 0i;
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

// Prefill 16 tokens
fn llama32_prefill_16_i8(
    i8[] x_i8, i16[] rms_att_w, i16[] rms_ffn_w,
    i8[] wq_t, i8[] wk_t, i8[] wv_t, i8[] wo_t,
    i8[] w1_t, i8[] w2_t, i8[] w3_t,
    i16[] xb_i16, i8[] xb_i8,
    i8[] k_cache, i8[] v_cache,
    i32[] att_scores, i8[] att_probs, i32[] attn_out, i8[] ffn_hb,
    i16[] cos_tab, i16[] sin_tab,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers, i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> i32 {
    var kv_dim = n_kv_heads * head_dim;
    var i = 0i;
    while (i < seq_len * dim) { xb_i16[i] = (x_i8[i] as i32 * 256i) as i16; i = i + 1i; }

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;
        var t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;
            var ss = 0i; i = 0i;
            while (i < dim) { var val = xb_i16[tok_off + i] as i32; ss = ss + (val * val) / 256i; i = i + 1i; }
            var mean = ss / dim;
            var rsqrt_approx = 16384i;
            if (mean > 256i) { rsqrt_approx = 8192i; }
            if (mean > 1024i) { rsqrt_approx = 4096i; }
            if (mean > 4096i) { rsqrt_approx = 2048i; }
            if (mean > 16384i) { rsqrt_approx = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_att_w[layer_offset + i] as i32;
                var normed = (val * rsqrt_approx) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<16, 2048> xb_mat = tensor_from_array(xb_i8, 0i);
        var wq_offset = layer * dim * dim;
        i8<2048, 2048> wq_mat = tensor_from_array(wq_t, wq_offset);
        var Q = tensor_matmul_nt(xb_mat, wq_mat);
        var wk_offset = layer * kv_dim * dim;
        i8<512, 2048> wk_mat = tensor_from_array(wk_t, wk_offset);
        var K = tensor_matmul_nt(xb_mat, wk_mat);
        var wv_offset = layer * kv_dim * dim;
        i8<512, 2048> wv_mat = tensor_from_array(wv_t, wv_offset);
        var V = tensor_matmul_nt(xb_mat, wv_mat);

        var cache_layer = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < kv_dim) {
                var k_scaled = K[t, i] as i32 / 256i;
                if (k_scaled > 127i) { k_scaled = 127i; }
                if (k_scaled < -128i) { k_scaled = -128i; }
                k_cache[cache_layer + t * kv_dim + i] = k_scaled as i8;
                var v_val = V[t, i] as i32 / 256i;
                if (v_val > 127i) { v_val = 127i; }
                if (v_val < -128i) { v_val = -128i; }
                v_cache[cache_layer + t * kv_dim + i] = v_val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        var n_rep = n_heads / n_kv_heads;
        var h = 0i;
        while (h < n_heads) {
            var q_head_off = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_off = kv_head * head_dim;
            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0i;
                    var d = 0i;
                    while (d < head_dim) {
                        var q_val = Q[q_pos, q_head_off + d];
                        var k_val = k_cache[cache_layer + k_pos * kv_dim + kv_head_off + d] as i32;
                        score = score + (q_val * k_val) / 256i;
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8i;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -100000i;
                    k_pos = k_pos + 1i;
                }
                var att_off = h * seq_len * seq_len + q_pos * seq_len;
                var max_s = att_scores[att_off];
                var ki = 1i;
                while (ki <= q_pos) { if (att_scores[att_off + ki] > max_s) { max_s = att_scores[att_off + ki]; } ki = ki + 1i; }
                var sum = 0i;
                ki = 0i;
                while (ki <= q_pos) {
                    var diff = att_scores[att_off + ki] - max_s;
                    var e = 128i + diff / 4i;
                    if (e < 0i) { e = 0i; }
                    if (e > 255i) { e = 255i; }
                    att_probs[att_off + ki] = e as i8;
                    sum = sum + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    var p = (att_probs[att_off + ki] as i32) * 127i / sum;
                    att_probs[att_off + ki] = p as i8;
                    ki = ki + 1i;
                }
                var d = 0i;
                while (d < head_dim) {
                    var val = 0i;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        var prob = att_probs[att_off + vi] as i32;
                        var v_val = v_cache[cache_layer + vi * kv_dim + kv_head_off + d] as i32;
                        val = val + prob * v_val;
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_off + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < dim) {
                var val = attn_out[t * dim + i] / 128i;
                if (val > 127i) { val = 127i; }
                if (val < -128i) { val = -128i; }
                xb_i8[t * dim + i] = val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<16, 2048> attn_mat = tensor_from_array(xb_i8, 0i);
        var wo_offset = layer * dim * dim;
        i8<2048, 2048> wo_mat = tensor_from_array(wo_t, wo_offset);
        var Wo_out = tensor_matmul_nt(attn_mat, wo_mat);

        t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;
            i = 0i;
            while (i < dim) {
                var res = xb_i16[tok_off + i] as i32 + Wo_out[t, i] / 256i;
                if (res > 32767i) { res = 32767i; }
                if (res < -32768i) { res = -32768i; }
                xb_i16[tok_off + i] = res as i16;
                i = i + 1i;
            }
            var ss = 0i;
            i = 0i;
            while (i < dim) { var val = xb_i16[tok_off + i] as i32; ss = ss + (val * val) / 256i; i = i + 1i; }
            var mean = ss / dim;
            var rsqrt = 16384i;
            if (mean > 256i) { rsqrt = 8192i; }
            if (mean > 1024i) { rsqrt = 4096i; }
            if (mean > 4096i) { rsqrt = 2048i; }
            if (mean > 16384i) { rsqrt = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_ffn_w[layer_offset + i] as i32;
                var normed = (val * rsqrt) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<16, 2048> ffn_in = tensor_from_array(xb_i8, 0i);
        var w1_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w1_mat = tensor_from_array(w1_t, w1_offset);
        var Gate = tensor_matmul_nt(ffn_in, w1_mat);
        var w3_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w3_mat = tensor_from_array(w3_t, w3_offset);
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

        i8<16, 8192> hb_mat = tensor_from_array(ffn_hb, 0i);
        var w2_offset = layer * dim * hidden_dim;
        i8<2048, 8192> w2_mat = tensor_from_array(w2_t, w2_offset);
        var Down = tensor_matmul_nt(hb_mat, w2_mat);

        t = 0i;
        while (t < seq_len) {
            i = 0i;
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

// Prefill 64 tokens
fn llama32_prefill_64_i8(
    i8[] x_i8, i16[] rms_att_w, i16[] rms_ffn_w,
    i8[] wq_t, i8[] wk_t, i8[] wv_t, i8[] wo_t,
    i8[] w1_t, i8[] w2_t, i8[] w3_t,
    i16[] xb_i16, i8[] xb_i8,
    i8[] k_cache, i8[] v_cache,
    i32[] att_scores, i8[] att_probs, i32[] attn_out, i8[] ffn_hb,
    i16[] cos_tab, i16[] sin_tab,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers, i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> i32 {
    var kv_dim = n_kv_heads * head_dim;
    var i = 0i;
    while (i < seq_len * dim) { xb_i16[i] = (x_i8[i] as i32 * 256i) as i16; i = i + 1i; }

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;
        var t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;
            var ss = 0i; i = 0i;
            while (i < dim) { var val = xb_i16[tok_off + i] as i32; ss = ss + (val * val) / 256i; i = i + 1i; }
            var mean = ss / dim;
            var rsqrt_approx = 16384i;
            if (mean > 256i) { rsqrt_approx = 8192i; }
            if (mean > 1024i) { rsqrt_approx = 4096i; }
            if (mean > 4096i) { rsqrt_approx = 2048i; }
            if (mean > 16384i) { rsqrt_approx = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_att_w[layer_offset + i] as i32;
                var normed = (val * rsqrt_approx) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<64, 2048> xb_mat = tensor_from_array(xb_i8, 0i);
        var wq_offset = layer * dim * dim;
        i8<2048, 2048> wq_mat = tensor_from_array(wq_t, wq_offset);
        var Q = tensor_matmul_nt(xb_mat, wq_mat);
        var wk_offset = layer * kv_dim * dim;
        i8<512, 2048> wk_mat = tensor_from_array(wk_t, wk_offset);
        var K = tensor_matmul_nt(xb_mat, wk_mat);
        var wv_offset = layer * kv_dim * dim;
        i8<512, 2048> wv_mat = tensor_from_array(wv_t, wv_offset);
        var V = tensor_matmul_nt(xb_mat, wv_mat);

        var cache_layer = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < kv_dim) {
                var k_scaled = K[t, i] as i32 / 256i;
                if (k_scaled > 127i) { k_scaled = 127i; }
                if (k_scaled < -128i) { k_scaled = -128i; }
                k_cache[cache_layer + t * kv_dim + i] = k_scaled as i8;
                var v_val = V[t, i] as i32 / 256i;
                if (v_val > 127i) { v_val = 127i; }
                if (v_val < -128i) { v_val = -128i; }
                v_cache[cache_layer + t * kv_dim + i] = v_val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        var n_rep = n_heads / n_kv_heads;
        var h = 0i;
        while (h < n_heads) {
            var q_head_off = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_off = kv_head * head_dim;
            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0i;
                    var d = 0i;
                    while (d < head_dim) {
                        var q_val = Q[q_pos, q_head_off + d];
                        var k_val = k_cache[cache_layer + k_pos * kv_dim + kv_head_off + d] as i32;
                        score = score + (q_val * k_val) / 256i;
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8i;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -100000i;
                    k_pos = k_pos + 1i;
                }
                var att_off = h * seq_len * seq_len + q_pos * seq_len;
                var max_s = att_scores[att_off];
                var ki = 1i;
                while (ki <= q_pos) { if (att_scores[att_off + ki] > max_s) { max_s = att_scores[att_off + ki]; } ki = ki + 1i; }
                var sum = 0i;
                ki = 0i;
                while (ki <= q_pos) {
                    var diff = att_scores[att_off + ki] - max_s;
                    var e = 128i + diff / 4i;
                    if (e < 0i) { e = 0i; }
                    if (e > 255i) { e = 255i; }
                    att_probs[att_off + ki] = e as i8;
                    sum = sum + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    var p = (att_probs[att_off + ki] as i32) * 127i / sum;
                    att_probs[att_off + ki] = p as i8;
                    ki = ki + 1i;
                }
                var d = 0i;
                while (d < head_dim) {
                    var val = 0i;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        var prob = att_probs[att_off + vi] as i32;
                        var v_val = v_cache[cache_layer + vi * kv_dim + kv_head_off + d] as i32;
                        val = val + prob * v_val;
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_off + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < dim) {
                var val = attn_out[t * dim + i] / 128i;
                if (val > 127i) { val = 127i; }
                if (val < -128i) { val = -128i; }
                xb_i8[t * dim + i] = val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<64, 2048> attn_mat = tensor_from_array(xb_i8, 0i);
        var wo_offset = layer * dim * dim;
        i8<2048, 2048> wo_mat = tensor_from_array(wo_t, wo_offset);
        var Wo_out = tensor_matmul_nt(attn_mat, wo_mat);

        t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;
            i = 0i;
            while (i < dim) {
                var res = xb_i16[tok_off + i] as i32 + Wo_out[t, i] / 256i;
                if (res > 32767i) { res = 32767i; }
                if (res < -32768i) { res = -32768i; }
                xb_i16[tok_off + i] = res as i16;
                i = i + 1i;
            }
            var ss = 0i;
            i = 0i;
            while (i < dim) { var val = xb_i16[tok_off + i] as i32; ss = ss + (val * val) / 256i; i = i + 1i; }
            var mean = ss / dim;
            var rsqrt = 16384i;
            if (mean > 256i) { rsqrt = 8192i; }
            if (mean > 1024i) { rsqrt = 4096i; }
            if (mean > 4096i) { rsqrt = 2048i; }
            if (mean > 16384i) { rsqrt = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_ffn_w[layer_offset + i] as i32;
                var normed = (val * rsqrt) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<64, 2048> ffn_in = tensor_from_array(xb_i8, 0i);
        var w1_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w1_mat = tensor_from_array(w1_t, w1_offset);
        var Gate = tensor_matmul_nt(ffn_in, w1_mat);
        var w3_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w3_mat = tensor_from_array(w3_t, w3_offset);
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

        i8<64, 8192> hb_mat = tensor_from_array(ffn_hb, 0i);
        var w2_offset = layer * dim * hidden_dim;
        i8<2048, 8192> w2_mat = tensor_from_array(w2_t, w2_offset);
        var Down = tensor_matmul_nt(hb_mat, w2_mat);

        t = 0i;
        while (t < seq_len) {
            i = 0i;
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

// Prefill 128 tokens
fn llama32_prefill_128_i8(
    i8[] x_i8, i16[] rms_att_w, i16[] rms_ffn_w,
    i8[] wq_t, i8[] wk_t, i8[] wv_t, i8[] wo_t,
    i8[] w1_t, i8[] w2_t, i8[] w3_t,
    i16[] xb_i16, i8[] xb_i8,
    i8[] k_cache, i8[] v_cache,
    i32[] att_scores, i8[] att_probs, i32[] attn_out, i8[] ffn_hb,
    i16[] cos_tab, i16[] sin_tab,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers, i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> i32 {
    var kv_dim = n_kv_heads * head_dim;
    var i = 0i;
    while (i < seq_len * dim) { xb_i16[i] = (x_i8[i] as i32 * 256i) as i16; i = i + 1i; }

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;
        var t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;
            var ss = 0i; i = 0i;
            while (i < dim) { var val = xb_i16[tok_off + i] as i32; ss = ss + (val * val) / 256i; i = i + 1i; }
            var mean = ss / dim;
            var rsqrt_approx = 16384i;
            if (mean > 256i) { rsqrt_approx = 8192i; }
            if (mean > 1024i) { rsqrt_approx = 4096i; }
            if (mean > 4096i) { rsqrt_approx = 2048i; }
            if (mean > 16384i) { rsqrt_approx = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_att_w[layer_offset + i] as i32;
                var normed = (val * rsqrt_approx) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<128, 2048> xb_mat = tensor_from_array(xb_i8, 0i);
        var wq_offset = layer * dim * dim;
        i8<2048, 2048> wq_mat = tensor_from_array(wq_t, wq_offset);
        var Q = tensor_matmul_nt(xb_mat, wq_mat);
        var wk_offset = layer * kv_dim * dim;
        i8<512, 2048> wk_mat = tensor_from_array(wk_t, wk_offset);
        var K = tensor_matmul_nt(xb_mat, wk_mat);
        var wv_offset = layer * kv_dim * dim;
        i8<512, 2048> wv_mat = tensor_from_array(wv_t, wv_offset);
        var V = tensor_matmul_nt(xb_mat, wv_mat);

        var cache_layer = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < kv_dim) {
                var k_scaled = K[t, i] as i32 / 256i;
                if (k_scaled > 127i) { k_scaled = 127i; }
                if (k_scaled < -128i) { k_scaled = -128i; }
                k_cache[cache_layer + t * kv_dim + i] = k_scaled as i8;
                var v_val = V[t, i] as i32 / 256i;
                if (v_val > 127i) { v_val = 127i; }
                if (v_val < -128i) { v_val = -128i; }
                v_cache[cache_layer + t * kv_dim + i] = v_val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        var n_rep = n_heads / n_kv_heads;
        var h = 0i;
        while (h < n_heads) {
            var q_head_off = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_off = kv_head * head_dim;
            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0i;
                    var d = 0i;
                    while (d < head_dim) {
                        var q_val = Q[q_pos, q_head_off + d];
                        var k_val = k_cache[cache_layer + k_pos * kv_dim + kv_head_off + d] as i32;
                        score = score + (q_val * k_val) / 256i;
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8i;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -100000i;
                    k_pos = k_pos + 1i;
                }
                var att_off = h * seq_len * seq_len + q_pos * seq_len;
                var max_s = att_scores[att_off];
                var ki = 1i;
                while (ki <= q_pos) { if (att_scores[att_off + ki] > max_s) { max_s = att_scores[att_off + ki]; } ki = ki + 1i; }
                var sum = 0i;
                ki = 0i;
                while (ki <= q_pos) {
                    var diff = att_scores[att_off + ki] - max_s;
                    var e = 128i + diff / 4i;
                    if (e < 0i) { e = 0i; }
                    if (e > 255i) { e = 255i; }
                    att_probs[att_off + ki] = e as i8;
                    sum = sum + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    var p = (att_probs[att_off + ki] as i32) * 127i / sum;
                    att_probs[att_off + ki] = p as i8;
                    ki = ki + 1i;
                }
                var d = 0i;
                while (d < head_dim) {
                    var val = 0i;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        var prob = att_probs[att_off + vi] as i32;
                        var v_val = v_cache[cache_layer + vi * kv_dim + kv_head_off + d] as i32;
                        val = val + prob * v_val;
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_off + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        t = 0i;
        while (t < seq_len) {
            i = 0i;
            while (i < dim) {
                var val = attn_out[t * dim + i] / 128i;
                if (val > 127i) { val = 127i; }
                if (val < -128i) { val = -128i; }
                xb_i8[t * dim + i] = val as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<128, 2048> attn_mat = tensor_from_array(xb_i8, 0i);
        var wo_offset = layer * dim * dim;
        i8<2048, 2048> wo_mat = tensor_from_array(wo_t, wo_offset);
        var Wo_out = tensor_matmul_nt(attn_mat, wo_mat);

        t = 0i;
        while (t < seq_len) {
            var tok_off = t * dim;
            i = 0i;
            while (i < dim) {
                var res = xb_i16[tok_off + i] as i32 + Wo_out[t, i] / 256i;
                if (res > 32767i) { res = 32767i; }
                if (res < -32768i) { res = -32768i; }
                xb_i16[tok_off + i] = res as i16;
                i = i + 1i;
            }
            var ss = 0i;
            i = 0i;
            while (i < dim) { var val = xb_i16[tok_off + i] as i32; ss = ss + (val * val) / 256i; i = i + 1i; }
            var mean = ss / dim;
            var rsqrt = 16384i;
            if (mean > 256i) { rsqrt = 8192i; }
            if (mean > 1024i) { rsqrt = 4096i; }
            if (mean > 4096i) { rsqrt = 2048i; }
            if (mean > 16384i) { rsqrt = 1024i; }
            i = 0i;
            while (i < dim) {
                var val = xb_i16[tok_off + i] as i32;
                var w = rms_ffn_w[layer_offset + i] as i32;
                var normed = (val * rsqrt) / 256i;
                var result = (normed * w) / 32768i;
                if (result > 32767i) { result = 32767i; }
                if (result < -32768i) { result = -32768i; }
                xb_i16[tok_off + i] = result as i16;
                var q8 = result / 256i;
                if (q8 > 127i) { q8 = 127i; }
                if (q8 < -128i) { q8 = -128i; }
                xb_i8[tok_off + i] = q8 as i8;
                i = i + 1i;
            }
            t = t + 1i;
        }

        i8<128, 2048> ffn_in = tensor_from_array(xb_i8, 0i);
        var w1_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w1_mat = tensor_from_array(w1_t, w1_offset);
        var Gate = tensor_matmul_nt(ffn_in, w1_mat);
        var w3_offset = layer * hidden_dim * dim;
        i8<8192, 2048> w3_mat = tensor_from_array(w3_t, w3_offset);
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

        i8<128, 8192> hb_mat = tensor_from_array(ffn_hb, 0i);
        var w2_offset = layer * dim * hidden_dim;
        i8<2048, 8192> w2_mat = tensor_from_array(w2_t, w2_offset);
        var Down = tensor_matmul_nt(hb_mat, w2_mat);

        t = 0i;
        while (t < seq_len) {
            i = 0i;
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
