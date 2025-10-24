// LLaMA 2 Transformer Forward Pass
// All arrays allocated by host, kernel only does computation

fn llama2_forward(
    // Weights (host-allocated)
    f32[] token_emb,
    f32[] rms_att_w,
    f32[] rms_ffn_w,
    f32[] wq, f32[] wk, f32[] wv, f32[] wo,
    f32[] w1, f32[] w2, f32[] w3,
    f32[] rms_final_w,
    f32[] wcls,
    // Activations (host-allocated buffers)
    f32[] x,
    f32[] xb,
    f32[] xb2,
    f32[] q, f32[] k, f32[] v,
    f32[] att,
    f32[] hb, f32[] hb2,
    f32[] logits,
    // KV cache (host-allocated)
    f32[] key_cache,
    f32[] value_cache,
    // Parameters
    i64 token, i64 pos,
    i64 dim, i64 hidden_dim,
    i64 n_layers, i64 n_heads, i64 n_kv_heads,
    i64 vocab_size, i64 seq_len
) -> f32 {
    var head_size = dim / n_heads;
    var kv_dim = (dim * n_kv_heads) / n_heads;
    var kv_mul = n_heads / n_kv_heads;

    // 1. Token embedding
    var token_offset = token * dim;
    var i = 0i;
    while (i < dim) {
        x[i] = token_emb[token_offset + i];
        i = i + 1i;
    }

    // 2. Transformer layers
    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        // === Attention ===

        // RMSNorm
        var j = 0i;
        while (j < dim) {
            xb[j] = 0.0;
            j = j + 1i;
        }

        // Use xb2 temporarily to store rms weights
        var rw = 0i;
        while (rw < dim) {
            xb2[rw] = rms_att_w[layer_offset + rw];
            rw = rw + 1i;
        }
        xb = rmsnorm(x, xb2, xb, dim, 0.00001);

        // QKV projections
        var qkv_i = 0i;
        while (qkv_i < dim) {
            q[qkv_i] = 0.0;
            qkv_i = qkv_i + 1i;
        }
        var kv_i = 0i;
        while (kv_i < kv_dim) {
            k[kv_i] = 0.0;
            v[kv_i] = 0.0;
            kv_i = kv_i + 1i;
        }

        // Weight offsets for this layer
        var wq_off = layer * dim * dim;
        var wk_off = layer * dim * kv_dim;
        var wv_off = layer * dim * kv_dim;
        var wo_off = layer * dim * dim;

        // Q projection: q = wq @ xb  (dim x dim) @ (dim x 1) -> (dim x 1)
        var qi = 0i;
        while (qi < dim) {
            var sum = 0.0;
            var qj = 0i;
            while (qj < dim) {
                sum = sum + wq[wq_off + qi * dim + qj] * xb[qj];
                qj = qj + 1i;
            }
            q[qi] = sum;
            qi = qi + 1i;
        }

        // K projection: k = wk @ xb  (kv_dim x dim) @ (dim x 1) -> (kv_dim x 1)
        var ki = 0i;
        while (ki < kv_dim) {
            var sum = 0.0;
            var kj = 0i;
            while (kj < dim) {
                sum = sum + wk[wk_off + ki * dim + kj] * xb[kj];
                kj = kj + 1i;
            }
            k[ki] = sum;
            ki = ki + 1i;
        }

        // V projection: v = wv @ xb  (kv_dim x dim) @ (dim x 1) -> (kv_dim x 1)
        var vi = 0i;
        while (vi < kv_dim) {
            var sum = 0.0;
            var vj = 0i;
            while (vj < dim) {
                sum = sum + wv[wv_off + vi * dim + vj] * xb[vj];
                vj = vj + 1i;
            }
            v[vi] = sum;
            vi = vi + 1i;
        }

        // Cache KV
        var kv_cache_off = layer * seq_len * kv_dim + pos * kv_dim;
        var c = 0i;
        while (c < kv_dim) {
            key_cache[kv_cache_off + c] = k[c];
            value_cache[kv_cache_off + c] = v[c];
            c = c + 1i;
        }

        // Multi-head attention
        var h = 0i;
        while (h < n_heads) {
            var q_off = h * head_size;
            var kv_head = h / kv_mul;
            var kv_off_base = kv_head * head_size;

            // Compute attention scores for all positions up to pos
            var att_off = h * seq_len;
            var t = 0i;
            while (t <= pos) {
                var key_pos_off = layer * seq_len * kv_dim + t * kv_dim + kv_off_base;

                // Dot product q @ k
                var score = 0.0;
                var hs = 0i;
                while (hs < head_size) {
                    score = score + q[q_off + hs] * key_cache[key_pos_off + hs];
                    hs = hs + 1i;
                }

                // Scale by 1/sqrt(head_size)
                score = score / 8.0;  // Approx sqrt(64)
                att[att_off + t] = score;
                t = t + 1i;
            }

            // Softmax over attention scores (positions 0 to pos)
            var att_len = pos + 1i;
            var att_tmp = array<f32>([seq_len]);
            var a_idx = 0i;
            while (a_idx < att_len) {
                att_tmp[a_idx] = att[att_off + a_idx];
                a_idx = a_idx + 1i;
            }
            var att_soft = array<f32>([seq_len]);
            var a_init = 0i;
            while (a_init < att_len) {
                att_soft[a_init] = 0.0;
                a_init = a_init + 1i;
            }
            att_soft = softmax(att_tmp, att_soft, att_len);

            // Weighted sum of values
            var hs2 = 0i;
            while (hs2 < head_size) {
                var val = 0.0;
                var t2 = 0i;
                while (t2 <= pos) {
                    var value_pos_off = layer * seq_len * kv_dim + t2 * kv_dim + kv_off_base + hs2;
                    val = val + att_soft[t2] * value_cache[value_pos_off];
                    t2 = t2 + 1i;
                }
                xb2[q_off + hs2] = val;
                hs2 = hs2 + 1i;
            }

            h = h + 1i;
        }

        // Output projection: xb = wo @ xb2  (dim x dim) @ (dim x 1) -> (dim x 1)
        var oi = 0i;
        while (oi < dim) {
            var sum = 0.0;
            var oj = 0i;
            while (oj < dim) {
                sum = sum + wo[wo_off + oi * dim + oj] * xb2[oj];
                oj = oj + 1i;
            }
            xb[oi] = sum;
            oi = oi + 1i;
        }

        // Residual
        var r1 = 0i;
        while (r1 < dim) {
            x[r1] = x[r1] + xb[r1];
            r1 = r1 + 1i;
        }

        // === FFN ===

        // RMSNorm
        var j2 = 0i;
        while (j2 < dim) {
            xb[j2] = 0.0;
            j2 = j2 + 1i;
        }

        // Use xb2 temporarily to store rms weights
        var rw2 = 0i;
        while (rw2 < dim) {
            xb2[rw2] = rms_ffn_w[layer_offset + rw2];
            rw2 = rw2 + 1i;
        }
        xb = rmsnorm(x, xb2, xb, dim, 0.00001);

        // SwiGLU FFN
        var w1_off = layer * hidden_dim * dim;
        var w2_off = layer * dim * hidden_dim;
        var w3_off = layer * hidden_dim * dim;

        // Gate projection: hb = w1 @ xb  (hidden_dim x dim) @ (dim x 1) -> (hidden_dim x 1)
        var g_init = 0i;
        while (g_init < hidden_dim) {
            hb[g_init] = 0.0;
            g_init = g_init + 1i;
        }
        var gi = 0i;
        while (gi < hidden_dim) {
            var sum = 0.0;
            var gj = 0i;
            while (gj < dim) {
                sum = sum + w1[w1_off + gi * dim + gj] * xb[gj];
                gj = gj + 1i;
            }
            hb[gi] = sum;
            gi = gi + 1i;
        }

        // Up projection: hb2 = w3 @ xb  (hidden_dim x dim) @ (dim x 1) -> (hidden_dim x 1)
        var u_init = 0i;
        while (u_init < hidden_dim) {
            hb2[u_init] = 0.0;
            u_init = u_init + 1i;
        }
        var ui = 0i;
        while (ui < hidden_dim) {
            var sum = 0.0;
            var uj = 0i;
            while (uj < dim) {
                sum = sum + w3[w3_off + ui * dim + uj] * xb[uj];
                uj = uj + 1i;
            }
            hb2[ui] = sum;
            ui = ui + 1i;
        }

        // SiLU activation on gate
        var hb_silu = array<f32>([hidden_dim]);
        var s_init = 0i;
        while (s_init < hidden_dim) {
            hb_silu[s_init] = 0.0;
            s_init = s_init + 1i;
        }
        hb_silu = silu(hb, hb_silu, hidden_dim);

        // Element-wise multiply: hb = hb_silu * hb2
        var m_idx = 0i;
        while (m_idx < hidden_dim) {
            hb[m_idx] = hb_silu[m_idx] * hb2[m_idx];
            m_idx = m_idx + 1i;
        }

        // Down projection: xb = w2 @ hb  (dim x hidden_dim) @ (hidden_dim x 1) -> (dim x 1)
        var d_init = 0i;
        while (d_init < dim) {
            xb[d_init] = 0.0;
            d_init = d_init + 1i;
        }
        var di = 0i;
        while (di < dim) {
            var sum = 0.0;
            var dj = 0i;
            while (dj < hidden_dim) {
                sum = sum + w2[w2_off + di * hidden_dim + dj] * hb[dj];
                dj = dj + 1i;
            }
            xb[di] = sum;
            di = di + 1i;
        }

        // Residual
        var r2 = 0i;
        while (r2 < dim) {
            x[r2] = x[r2] + xb[r2];
            r2 = r2 + 1i;
        }

        layer = layer + 1i;
    }

    // 3. Final norm
    var f = 0i;
    while (f < dim) {
        xb[f] = 0.0;
        f = f + 1i;
    }
    xb = rmsnorm(x, rms_final_w, xb, dim, 0.00001);

    // 4. Classifier: logits = wcls @ xb  (vocab_size x dim) @ (dim x 1) -> (vocab_size x 1)
    var l_init = 0i;
    while (l_init < vocab_size) {
        logits[l_init] = 0.0;
        l_init = l_init + 1i;
    }
    var li = 0i;
    while (li < vocab_size) {
        var sum = 0.0;
        var lj = 0i;
        while (lj < dim) {
            sum = sum + wcls[li * dim + lj] * xb[lj];
            lj = lj + 1i;
        }
        logits[li] = sum;
        li = li + 1i;
    }

    return logits[0i];
}
