// LLaMA 3.2-1B FP32 Prefill - Multiple sequence length variants
// Uses tensor_matmul for FP32 matmuls
//
// Architecture:
//   dim = 2048, n_layers = 16, n_heads = 32, n_kv_heads = 8
//   head_dim = 64, hidden_dim = 8192

// ============================================================================
// Prefill 8 tokens
// ============================================================================
fn llama32_fp32_prefill_8(
    f32[] x, f32[] rms_att_w, f32[] rms_ffn_w, f32[] rms_final_w,
    f32[] wq, f32[] wk, f32[] wv, f32[] wo,
    f32[] w1, f32[] w2, f32[] w3,
    f32[] xb, f32[] q_buf, f32[] k_buf, f32[] v_buf,
    f32[] k_cache, f32[] v_cache,
    f32[] att_scores, f32[] att_probs, f32[] attn_out, f32[] hb,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers, i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> f32 {
    var kv_dim = n_kv_heads * head_dim;
    var n_rep = n_heads / n_kv_heads;

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        // RMSNorm for all tokens
        var t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0.0;
            var i = 0i;
            while (i < dim) {
                var val = x[token_offset + i];
                ss = ss + val * val;
                i = i + 1i;
            }
            ss = 1.0 / sqrt(ss / (dim * 1.0) + 0.00001);
            i = 0i;
            while (i < dim) {
                xb[token_offset + i] = x[token_offset + i] * ss * rms_att_w[layer_offset + i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        // QKV projections using tensor_matmul
        f32<8, 2048> xb_mat = tensor_from_array(xb, 0i);
        var wq_offset = layer * dim * dim;
        f32<2048, 2048> wq_mat = tensor_from_array(wq, wq_offset);
        var Q = tensor_matmul(xb_mat, wq_mat);

        var wk_offset = layer * kv_dim * dim;
        f32<2048, 512> wk_mat = tensor_from_array(wk, wk_offset);
        var K = tensor_matmul(xb_mat, wk_mat);

        var wv_offset = layer * kv_dim * dim;
        f32<2048, 512> wv_mat = tensor_from_array(wv, wv_offset);
        var V = tensor_matmul(xb_mat, wv_mat);

        // Store KV cache
        var cache_layer_offset = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < kv_dim) {
                k_cache[cache_layer_offset + t * kv_dim + i] = K[t, i];
                v_cache[cache_layer_offset + t * kv_dim + i] = V[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        // Causal attention
        var h = 0i;
        while (h < n_heads) {
            var q_head_offset = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_offset = kv_head * head_dim;

            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0.0;
                    var d = 0i;
                    while (d < head_dim) {
                        score = score + Q[q_pos, q_head_offset + d] * k_cache[cache_layer_offset + k_pos * kv_dim + kv_head_offset + d];
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8.0;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -1000000.0;
                    k_pos = k_pos + 1i;
                }

                var att_offset = h * seq_len * seq_len + q_pos * seq_len;
                var max_val = att_scores[att_offset];
                var ki = 1i;
                while (ki <= q_pos) {
                    if (att_scores[att_offset + ki] > max_val) { max_val = att_scores[att_offset + ki]; }
                    ki = ki + 1i;
                }
                var sum_exp = 0.0;
                ki = 0i;
                while (ki <= q_pos) {
                    var e = exp(att_scores[att_offset + ki] - max_val);
                    att_probs[att_offset + ki] = e;
                    sum_exp = sum_exp + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    att_probs[att_offset + ki] = att_probs[att_offset + ki] / sum_exp;
                    ki = ki + 1i;
                }

                var d = 0i;
                while (d < head_dim) {
                    var val = 0.0;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        val = val + att_probs[att_offset + vi] * v_cache[cache_layer_offset + vi * kv_dim + kv_head_offset + d];
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_offset + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        // Output projection
        f32<8, 2048> attn_mat = tensor_from_array(attn_out, 0i);
        var wo_offset = layer * dim * dim;
        f32<2048, 2048> wo_mat = tensor_from_array(wo, wo_offset);
        var Wo_out = tensor_matmul(attn_mat, wo_mat);

        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                x[t * dim + i] = x[t * dim + i] + Wo_out[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        // FFN RMSNorm
        t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0.0;
            var i = 0i;
            while (i < dim) {
                var val = x[token_offset + i];
                ss = ss + val * val;
                i = i + 1i;
            }
            ss = 1.0 / sqrt(ss / (dim * 1.0) + 0.00001);
            i = 0i;
            while (i < dim) {
                xb[token_offset + i] = x[token_offset + i] * ss * rms_ffn_w[layer_offset + i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        // FFN
        var w1_offset = layer * hidden_dim * dim;
        var w3_offset = layer * hidden_dim * dim;
        var w2_offset = layer * dim * hidden_dim;

        f32<8, 2048> ffn_in = tensor_from_array(xb, 0i);
        f32<2048, 8192> w1_mat = tensor_from_array(w1, w1_offset);
        var Gate = tensor_matmul(ffn_in, w1_mat);

        f32<2048, 8192> w3_mat = tensor_from_array(w3, w3_offset);
        var Up = tensor_matmul(ffn_in, w3_mat);

        // SiLU(gate) * up
        t = 0i;
        while (t < seq_len) {
            var hi = 0i;
            while (hi < hidden_dim) {
                var gate_f = Gate[t, hi];
                var silu = gate_f / (1.0 + exp(0.0 - gate_f));
                hb[t * hidden_dim + hi] = silu * Up[t, hi];
                hi = hi + 1i;
            }
            t = t + 1i;
        }

        f32<8, 8192> hb_mat = tensor_from_array(hb, 0i);
        f32<8192, 2048> w2_mat = tensor_from_array(w2, w2_offset);
        var Down = tensor_matmul(hb_mat, w2_mat);

        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                x[t * dim + i] = x[t * dim + i] + Down[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        layer = layer + 1i;
    }
    return x[(seq_len - 1i) * dim];
}

// ============================================================================
// Prefill 16 tokens
// ============================================================================
fn llama32_fp32_prefill_16(
    f32[] x, f32[] rms_att_w, f32[] rms_ffn_w, f32[] rms_final_w,
    f32[] wq, f32[] wk, f32[] wv, f32[] wo,
    f32[] w1, f32[] w2, f32[] w3,
    f32[] xb, f32[] q_buf, f32[] k_buf, f32[] v_buf,
    f32[] k_cache, f32[] v_cache,
    f32[] att_scores, f32[] att_probs, f32[] attn_out, f32[] hb,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers, i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> f32 {
    var kv_dim = n_kv_heads * head_dim;
    var n_rep = n_heads / n_kv_heads;

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        var t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0.0;
            var i = 0i;
            while (i < dim) {
                var val = x[token_offset + i];
                ss = ss + val * val;
                i = i + 1i;
            }
            ss = 1.0 / sqrt(ss / (dim * 1.0) + 0.00001);
            i = 0i;
            while (i < dim) {
                xb[token_offset + i] = x[token_offset + i] * ss * rms_att_w[layer_offset + i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        f32<16, 2048> xb_mat = tensor_from_array(xb, 0i);
        var wq_offset = layer * dim * dim;
        f32<2048, 2048> wq_mat = tensor_from_array(wq, wq_offset);
        var Q = tensor_matmul(xb_mat, wq_mat);

        var wk_offset = layer * kv_dim * dim;
        f32<2048, 512> wk_mat = tensor_from_array(wk, wk_offset);
        var K = tensor_matmul(xb_mat, wk_mat);

        var wv_offset = layer * kv_dim * dim;
        f32<2048, 512> wv_mat = tensor_from_array(wv, wv_offset);
        var V = tensor_matmul(xb_mat, wv_mat);

        var cache_layer_offset = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < kv_dim) {
                k_cache[cache_layer_offset + t * kv_dim + i] = K[t, i];
                v_cache[cache_layer_offset + t * kv_dim + i] = V[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        var h = 0i;
        while (h < n_heads) {
            var q_head_offset = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_offset = kv_head * head_dim;

            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0.0;
                    var d = 0i;
                    while (d < head_dim) {
                        score = score + Q[q_pos, q_head_offset + d] * k_cache[cache_layer_offset + k_pos * kv_dim + kv_head_offset + d];
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8.0;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -1000000.0;
                    k_pos = k_pos + 1i;
                }

                var att_offset = h * seq_len * seq_len + q_pos * seq_len;
                var max_val = att_scores[att_offset];
                var ki = 1i;
                while (ki <= q_pos) {
                    if (att_scores[att_offset + ki] > max_val) { max_val = att_scores[att_offset + ki]; }
                    ki = ki + 1i;
                }
                var sum_exp = 0.0;
                ki = 0i;
                while (ki <= q_pos) {
                    var e = exp(att_scores[att_offset + ki] - max_val);
                    att_probs[att_offset + ki] = e;
                    sum_exp = sum_exp + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    att_probs[att_offset + ki] = att_probs[att_offset + ki] / sum_exp;
                    ki = ki + 1i;
                }

                var d = 0i;
                while (d < head_dim) {
                    var val = 0.0;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        val = val + att_probs[att_offset + vi] * v_cache[cache_layer_offset + vi * kv_dim + kv_head_offset + d];
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_offset + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        f32<16, 2048> attn_mat = tensor_from_array(attn_out, 0i);
        var wo_offset = layer * dim * dim;
        f32<2048, 2048> wo_mat = tensor_from_array(wo, wo_offset);
        var Wo_out = tensor_matmul(attn_mat, wo_mat);

        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                x[t * dim + i] = x[t * dim + i] + Wo_out[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0.0;
            var i = 0i;
            while (i < dim) {
                var val = x[token_offset + i];
                ss = ss + val * val;
                i = i + 1i;
            }
            ss = 1.0 / sqrt(ss / (dim * 1.0) + 0.00001);
            i = 0i;
            while (i < dim) {
                xb[token_offset + i] = x[token_offset + i] * ss * rms_ffn_w[layer_offset + i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        var w1_offset = layer * hidden_dim * dim;
        var w3_offset = layer * hidden_dim * dim;
        var w2_offset = layer * dim * hidden_dim;

        f32<16, 2048> ffn_in = tensor_from_array(xb, 0i);
        f32<2048, 8192> w1_mat = tensor_from_array(w1, w1_offset);
        var Gate = tensor_matmul(ffn_in, w1_mat);

        f32<2048, 8192> w3_mat = tensor_from_array(w3, w3_offset);
        var Up = tensor_matmul(ffn_in, w3_mat);

        t = 0i;
        while (t < seq_len) {
            var hi = 0i;
            while (hi < hidden_dim) {
                var gate_f = Gate[t, hi];
                var silu = gate_f / (1.0 + exp(0.0 - gate_f));
                hb[t * hidden_dim + hi] = silu * Up[t, hi];
                hi = hi + 1i;
            }
            t = t + 1i;
        }

        f32<16, 8192> hb_mat = tensor_from_array(hb, 0i);
        f32<8192, 2048> w2_mat = tensor_from_array(w2, w2_offset);
        var Down = tensor_matmul(hb_mat, w2_mat);

        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                x[t * dim + i] = x[t * dim + i] + Down[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        layer = layer + 1i;
    }
    return x[(seq_len - 1i) * dim];
}

// ============================================================================
// Prefill 32 tokens
// ============================================================================
fn llama32_fp32_prefill_32(
    f32[] x, f32[] rms_att_w, f32[] rms_ffn_w, f32[] rms_final_w,
    f32[] wq, f32[] wk, f32[] wv, f32[] wo,
    f32[] w1, f32[] w2, f32[] w3,
    f32[] xb, f32[] q_buf, f32[] k_buf, f32[] v_buf,
    f32[] k_cache, f32[] v_cache,
    f32[] att_scores, f32[] att_probs, f32[] attn_out, f32[] hb,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers, i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> f32 {
    var kv_dim = n_kv_heads * head_dim;
    var n_rep = n_heads / n_kv_heads;

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        var t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0.0;
            var i = 0i;
            while (i < dim) {
                var val = x[token_offset + i];
                ss = ss + val * val;
                i = i + 1i;
            }
            ss = 1.0 / sqrt(ss / (dim * 1.0) + 0.00001);
            i = 0i;
            while (i < dim) {
                xb[token_offset + i] = x[token_offset + i] * ss * rms_att_w[layer_offset + i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        f32<32, 2048> xb_mat = tensor_from_array(xb, 0i);
        var wq_offset = layer * dim * dim;
        f32<2048, 2048> wq_mat = tensor_from_array(wq, wq_offset);
        var Q = tensor_matmul(xb_mat, wq_mat);

        var wk_offset = layer * kv_dim * dim;
        f32<2048, 512> wk_mat = tensor_from_array(wk, wk_offset);
        var K = tensor_matmul(xb_mat, wk_mat);

        var wv_offset = layer * kv_dim * dim;
        f32<2048, 512> wv_mat = tensor_from_array(wv, wv_offset);
        var V = tensor_matmul(xb_mat, wv_mat);

        var cache_layer_offset = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < kv_dim) {
                k_cache[cache_layer_offset + t * kv_dim + i] = K[t, i];
                v_cache[cache_layer_offset + t * kv_dim + i] = V[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        var h = 0i;
        while (h < n_heads) {
            var q_head_offset = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_offset = kv_head * head_dim;

            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0.0;
                    var d = 0i;
                    while (d < head_dim) {
                        score = score + Q[q_pos, q_head_offset + d] * k_cache[cache_layer_offset + k_pos * kv_dim + kv_head_offset + d];
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8.0;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -1000000.0;
                    k_pos = k_pos + 1i;
                }

                var att_offset = h * seq_len * seq_len + q_pos * seq_len;
                var max_val = att_scores[att_offset];
                var ki = 1i;
                while (ki <= q_pos) {
                    if (att_scores[att_offset + ki] > max_val) { max_val = att_scores[att_offset + ki]; }
                    ki = ki + 1i;
                }
                var sum_exp = 0.0;
                ki = 0i;
                while (ki <= q_pos) {
                    var e = exp(att_scores[att_offset + ki] - max_val);
                    att_probs[att_offset + ki] = e;
                    sum_exp = sum_exp + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    att_probs[att_offset + ki] = att_probs[att_offset + ki] / sum_exp;
                    ki = ki + 1i;
                }

                var d = 0i;
                while (d < head_dim) {
                    var val = 0.0;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        val = val + att_probs[att_offset + vi] * v_cache[cache_layer_offset + vi * kv_dim + kv_head_offset + d];
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_offset + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        f32<32, 2048> attn_mat = tensor_from_array(attn_out, 0i);
        var wo_offset = layer * dim * dim;
        f32<2048, 2048> wo_mat = tensor_from_array(wo, wo_offset);
        var Wo_out = tensor_matmul(attn_mat, wo_mat);

        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                x[t * dim + i] = x[t * dim + i] + Wo_out[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0.0;
            var i = 0i;
            while (i < dim) {
                var val = x[token_offset + i];
                ss = ss + val * val;
                i = i + 1i;
            }
            ss = 1.0 / sqrt(ss / (dim * 1.0) + 0.00001);
            i = 0i;
            while (i < dim) {
                xb[token_offset + i] = x[token_offset + i] * ss * rms_ffn_w[layer_offset + i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        var w1_offset = layer * hidden_dim * dim;
        var w3_offset = layer * hidden_dim * dim;
        var w2_offset = layer * dim * hidden_dim;

        f32<32, 2048> ffn_in = tensor_from_array(xb, 0i);
        f32<2048, 8192> w1_mat = tensor_from_array(w1, w1_offset);
        var Gate = tensor_matmul(ffn_in, w1_mat);

        f32<2048, 8192> w3_mat = tensor_from_array(w3, w3_offset);
        var Up = tensor_matmul(ffn_in, w3_mat);

        t = 0i;
        while (t < seq_len) {
            var hi = 0i;
            while (hi < hidden_dim) {
                var gate_f = Gate[t, hi];
                var silu = gate_f / (1.0 + exp(0.0 - gate_f));
                hb[t * hidden_dim + hi] = silu * Up[t, hi];
                hi = hi + 1i;
            }
            t = t + 1i;
        }

        f32<32, 8192> hb_mat = tensor_from_array(hb, 0i);
        f32<8192, 2048> w2_mat = tensor_from_array(w2, w2_offset);
        var Down = tensor_matmul(hb_mat, w2_mat);

        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                x[t * dim + i] = x[t * dim + i] + Down[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        layer = layer + 1i;
    }
    return x[(seq_len - 1i) * dim];
}

// ============================================================================
// Prefill 64 tokens
// ============================================================================
fn llama32_fp32_prefill_64(
    f32[] x, f32[] rms_att_w, f32[] rms_ffn_w, f32[] rms_final_w,
    f32[] wq, f32[] wk, f32[] wv, f32[] wo,
    f32[] w1, f32[] w2, f32[] w3,
    f32[] xb, f32[] q_buf, f32[] k_buf, f32[] v_buf,
    f32[] k_cache, f32[] v_cache,
    f32[] att_scores, f32[] att_probs, f32[] attn_out, f32[] hb,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers, i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> f32 {
    var kv_dim = n_kv_heads * head_dim;
    var n_rep = n_heads / n_kv_heads;

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        var t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0.0;
            var i = 0i;
            while (i < dim) {
                var val = x[token_offset + i];
                ss = ss + val * val;
                i = i + 1i;
            }
            ss = 1.0 / sqrt(ss / (dim * 1.0) + 0.00001);
            i = 0i;
            while (i < dim) {
                xb[token_offset + i] = x[token_offset + i] * ss * rms_att_w[layer_offset + i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        f32<64, 2048> xb_mat = tensor_from_array(xb, 0i);
        var wq_offset = layer * dim * dim;
        f32<2048, 2048> wq_mat = tensor_from_array(wq, wq_offset);
        var Q = tensor_matmul(xb_mat, wq_mat);

        var wk_offset = layer * kv_dim * dim;
        f32<2048, 512> wk_mat = tensor_from_array(wk, wk_offset);
        var K = tensor_matmul(xb_mat, wk_mat);

        var wv_offset = layer * kv_dim * dim;
        f32<2048, 512> wv_mat = tensor_from_array(wv, wv_offset);
        var V = tensor_matmul(xb_mat, wv_mat);

        var cache_layer_offset = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < kv_dim) {
                k_cache[cache_layer_offset + t * kv_dim + i] = K[t, i];
                v_cache[cache_layer_offset + t * kv_dim + i] = V[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        var h = 0i;
        while (h < n_heads) {
            var q_head_offset = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_offset = kv_head * head_dim;

            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0.0;
                    var d = 0i;
                    while (d < head_dim) {
                        score = score + Q[q_pos, q_head_offset + d] * k_cache[cache_layer_offset + k_pos * kv_dim + kv_head_offset + d];
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8.0;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -1000000.0;
                    k_pos = k_pos + 1i;
                }

                var att_offset = h * seq_len * seq_len + q_pos * seq_len;
                var max_val = att_scores[att_offset];
                var ki = 1i;
                while (ki <= q_pos) {
                    if (att_scores[att_offset + ki] > max_val) { max_val = att_scores[att_offset + ki]; }
                    ki = ki + 1i;
                }
                var sum_exp = 0.0;
                ki = 0i;
                while (ki <= q_pos) {
                    var e = exp(att_scores[att_offset + ki] - max_val);
                    att_probs[att_offset + ki] = e;
                    sum_exp = sum_exp + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    att_probs[att_offset + ki] = att_probs[att_offset + ki] / sum_exp;
                    ki = ki + 1i;
                }

                var d = 0i;
                while (d < head_dim) {
                    var val = 0.0;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        val = val + att_probs[att_offset + vi] * v_cache[cache_layer_offset + vi * kv_dim + kv_head_offset + d];
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_offset + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        f32<64, 2048> attn_mat = tensor_from_array(attn_out, 0i);
        var wo_offset = layer * dim * dim;
        f32<2048, 2048> wo_mat = tensor_from_array(wo, wo_offset);
        var Wo_out = tensor_matmul(attn_mat, wo_mat);

        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                x[t * dim + i] = x[t * dim + i] + Wo_out[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0.0;
            var i = 0i;
            while (i < dim) {
                var val = x[token_offset + i];
                ss = ss + val * val;
                i = i + 1i;
            }
            ss = 1.0 / sqrt(ss / (dim * 1.0) + 0.00001);
            i = 0i;
            while (i < dim) {
                xb[token_offset + i] = x[token_offset + i] * ss * rms_ffn_w[layer_offset + i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        var w1_offset = layer * hidden_dim * dim;
        var w3_offset = layer * hidden_dim * dim;
        var w2_offset = layer * dim * hidden_dim;

        f32<64, 2048> ffn_in = tensor_from_array(xb, 0i);
        f32<2048, 8192> w1_mat = tensor_from_array(w1, w1_offset);
        var Gate = tensor_matmul(ffn_in, w1_mat);

        f32<2048, 8192> w3_mat = tensor_from_array(w3, w3_offset);
        var Up = tensor_matmul(ffn_in, w3_mat);

        t = 0i;
        while (t < seq_len) {
            var hi = 0i;
            while (hi < hidden_dim) {
                var gate_f = Gate[t, hi];
                var silu = gate_f / (1.0 + exp(0.0 - gate_f));
                hb[t * hidden_dim + hi] = silu * Up[t, hi];
                hi = hi + 1i;
            }
            t = t + 1i;
        }

        f32<64, 8192> hb_mat = tensor_from_array(hb, 0i);
        f32<8192, 2048> w2_mat = tensor_from_array(w2, w2_offset);
        var Down = tensor_matmul(hb_mat, w2_mat);

        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                x[t * dim + i] = x[t * dim + i] + Down[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        layer = layer + 1i;
    }
    return x[(seq_len - 1i) * dim];
}

// ============================================================================
// Prefill 128 tokens
// ============================================================================
fn llama32_fp32_prefill_128(
    f32[] x, f32[] rms_att_w, f32[] rms_ffn_w, f32[] rms_final_w,
    f32[] wq, f32[] wk, f32[] wv, f32[] wo,
    f32[] w1, f32[] w2, f32[] w3,
    f32[] xb, f32[] q_buf, f32[] k_buf, f32[] v_buf,
    f32[] k_cache, f32[] v_cache,
    f32[] att_scores, f32[] att_probs, f32[] attn_out, f32[] hb,
    i64 seq_len, i64 dim, i64 hidden_dim, i64 n_layers, i64 n_heads, i64 n_kv_heads, i64 head_dim
) -> f32 {
    var kv_dim = n_kv_heads * head_dim;
    var n_rep = n_heads / n_kv_heads;

    var layer = 0i;
    while (layer < n_layers) {
        var layer_offset = layer * dim;

        var t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0.0;
            var i = 0i;
            while (i < dim) {
                var val = x[token_offset + i];
                ss = ss + val * val;
                i = i + 1i;
            }
            ss = 1.0 / sqrt(ss / (dim * 1.0) + 0.00001);
            i = 0i;
            while (i < dim) {
                xb[token_offset + i] = x[token_offset + i] * ss * rms_att_w[layer_offset + i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        f32<128, 2048> xb_mat = tensor_from_array(xb, 0i);
        var wq_offset = layer * dim * dim;
        f32<2048, 2048> wq_mat = tensor_from_array(wq, wq_offset);
        var Q = tensor_matmul(xb_mat, wq_mat);

        var wk_offset = layer * kv_dim * dim;
        f32<2048, 512> wk_mat = tensor_from_array(wk, wk_offset);
        var K = tensor_matmul(xb_mat, wk_mat);

        var wv_offset = layer * kv_dim * dim;
        f32<2048, 512> wv_mat = tensor_from_array(wv, wv_offset);
        var V = tensor_matmul(xb_mat, wv_mat);

        var cache_layer_offset = layer * seq_len * kv_dim;
        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < kv_dim) {
                k_cache[cache_layer_offset + t * kv_dim + i] = K[t, i];
                v_cache[cache_layer_offset + t * kv_dim + i] = V[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        var h = 0i;
        while (h < n_heads) {
            var q_head_offset = h * head_dim;
            var kv_head = h / n_rep;
            var kv_head_offset = kv_head * head_dim;

            var q_pos = 0i;
            while (q_pos < seq_len) {
                var k_pos = 0i;
                while (k_pos <= q_pos) {
                    var score = 0.0;
                    var d = 0i;
                    while (d < head_dim) {
                        score = score + Q[q_pos, q_head_offset + d] * k_cache[cache_layer_offset + k_pos * kv_dim + kv_head_offset + d];
                        d = d + 1i;
                    }
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = score / 8.0;
                    k_pos = k_pos + 1i;
                }
                while (k_pos < seq_len) {
                    att_scores[h * seq_len * seq_len + q_pos * seq_len + k_pos] = -1000000.0;
                    k_pos = k_pos + 1i;
                }

                var att_offset = h * seq_len * seq_len + q_pos * seq_len;
                var max_val = att_scores[att_offset];
                var ki = 1i;
                while (ki <= q_pos) {
                    if (att_scores[att_offset + ki] > max_val) { max_val = att_scores[att_offset + ki]; }
                    ki = ki + 1i;
                }
                var sum_exp = 0.0;
                ki = 0i;
                while (ki <= q_pos) {
                    var e = exp(att_scores[att_offset + ki] - max_val);
                    att_probs[att_offset + ki] = e;
                    sum_exp = sum_exp + e;
                    ki = ki + 1i;
                }
                ki = 0i;
                while (ki <= q_pos) {
                    att_probs[att_offset + ki] = att_probs[att_offset + ki] / sum_exp;
                    ki = ki + 1i;
                }

                var d = 0i;
                while (d < head_dim) {
                    var val = 0.0;
                    var vi = 0i;
                    while (vi <= q_pos) {
                        val = val + att_probs[att_offset + vi] * v_cache[cache_layer_offset + vi * kv_dim + kv_head_offset + d];
                        vi = vi + 1i;
                    }
                    attn_out[q_pos * dim + q_head_offset + d] = val;
                    d = d + 1i;
                }
                q_pos = q_pos + 1i;
            }
            h = h + 1i;
        }

        f32<128, 2048> attn_mat = tensor_from_array(attn_out, 0i);
        var wo_offset = layer * dim * dim;
        f32<2048, 2048> wo_mat = tensor_from_array(wo, wo_offset);
        var Wo_out = tensor_matmul(attn_mat, wo_mat);

        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                x[t * dim + i] = x[t * dim + i] + Wo_out[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        t = 0i;
        while (t < seq_len) {
            var token_offset = t * dim;
            var ss = 0.0;
            var i = 0i;
            while (i < dim) {
                var val = x[token_offset + i];
                ss = ss + val * val;
                i = i + 1i;
            }
            ss = 1.0 / sqrt(ss / (dim * 1.0) + 0.00001);
            i = 0i;
            while (i < dim) {
                xb[token_offset + i] = x[token_offset + i] * ss * rms_ffn_w[layer_offset + i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        var w1_offset = layer * hidden_dim * dim;
        var w3_offset = layer * hidden_dim * dim;
        var w2_offset = layer * dim * hidden_dim;

        f32<128, 2048> ffn_in = tensor_from_array(xb, 0i);
        f32<2048, 8192> w1_mat = tensor_from_array(w1, w1_offset);
        var Gate = tensor_matmul(ffn_in, w1_mat);

        f32<2048, 8192> w3_mat = tensor_from_array(w3, w3_offset);
        var Up = tensor_matmul(ffn_in, w3_mat);

        t = 0i;
        while (t < seq_len) {
            var hi = 0i;
            while (hi < hidden_dim) {
                var gate_f = Gate[t, hi];
                var silu = gate_f / (1.0 + exp(0.0 - gate_f));
                hb[t * hidden_dim + hi] = silu * Up[t, hi];
                hi = hi + 1i;
            }
            t = t + 1i;
        }

        f32<128, 8192> hb_mat = tensor_from_array(hb, 0i);
        f32<8192, 2048> w2_mat = tensor_from_array(w2, w2_offset);
        var Down = tensor_matmul(hb_mat, w2_mat);

        t = 0i;
        while (t < seq_len) {
            var i = 0i;
            while (i < dim) {
                x[t * dim + i] = x[t * dim + i] + Down[t, i];
                i = i + 1i;
            }
            t = t + 1i;
        }

        layer = layer + 1i;
    }
    return x[(seq_len - 1i) * dim];
}
