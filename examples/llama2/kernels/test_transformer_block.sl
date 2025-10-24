// Mini Transformer Block Test
// Combines: RMSNorm -> Attention -> RMSNorm -> FFN (SwiGLU)

fn test_transformer_block() -> f32 {
    var dim = 4i;

    // Input hidden state
    var x = array<f32>([4]);
    x[0i] = 1.0;
    x[1i] = 2.0;
    x[2i] = 3.0;
    x[3i] = 4.0;

    // ===== Attention Block =====

    // 1. Pre-attention RMSNorm
    var attn_norm_weight = array<f32>([4]);
    attn_norm_weight[0i] = 1.0;
    attn_norm_weight[1i] = 1.0;
    attn_norm_weight[2i] = 1.0;
    attn_norm_weight[3i] = 1.0;

    var attn_norm_out = array<f32>([4]);
    attn_norm_out[0i] = 0.0;
    attn_norm_out[1i] = 0.0;
    attn_norm_out[2i] = 0.0;
    attn_norm_out[3i] = 0.0;

    var x_normed = rmsnorm(x, attn_norm_weight, attn_norm_out, dim, 0.00001);

    // 2. Simple attention (using normalized input as Q, K, V)
    // For simplicity, compute attention score for just the first element
    var attn_scores = array<f32>([2]);
    attn_scores[0i] = x_normed[0i] * x_normed[0i] + x_normed[1i] * x_normed[1i];  // self-attention score
    attn_scores[1i] = 0.0;  // padding

    var attn_weights = array<f32>([2]);
    attn_weights[0i] = 0.0;
    attn_weights[1i] = 0.0;

    var attn_out_temp = softmax(attn_scores, attn_weights, 2i);

    // 3. Attention output (simplified: just scale the input)
    var attn_out = array<f32>([4]);
    attn_out[0i] = x_normed[0i] * attn_out_temp[0i];
    attn_out[1i] = x_normed[1i] * attn_out_temp[0i];
    attn_out[2i] = x_normed[2i] * attn_out_temp[0i];
    attn_out[3i] = x_normed[3i] * attn_out_temp[0i];

    // 4. Residual connection
    var post_attn = array<f32>([4]);
    post_attn[0i] = x[0i] + attn_out[0i];
    post_attn[1i] = x[1i] + attn_out[1i];
    post_attn[2i] = x[2i] + attn_out[2i];
    post_attn[3i] = x[3i] + attn_out[3i];

    // ===== FFN Block =====

    // 5. Pre-FFN RMSNorm
    var ffn_norm_weight = array<f32>([4]);
    ffn_norm_weight[0i] = 1.0;
    ffn_norm_weight[1i] = 1.0;
    ffn_norm_weight[2i] = 1.0;
    ffn_norm_weight[3i] = 1.0;

    var ffn_norm_out = array<f32>([4]);
    ffn_norm_out[0i] = 0.0;
    ffn_norm_out[1i] = 0.0;
    ffn_norm_out[2i] = 0.0;
    ffn_norm_out[3i] = 0.0;

    var ffn_input = rmsnorm(post_attn, ffn_norm_weight, ffn_norm_out, dim, 0.00001);

    // 6. SwiGLU FFN
    var gate = array<f32>([4]);
    gate[0i] = ffn_input[0i] * 0.5;
    gate[1i] = ffn_input[1i] * 0.5;
    gate[2i] = ffn_input[2i] * 0.5;
    gate[3i] = ffn_input[3i] * 0.5;

    var up = array<f32>([4]);
    up[0i] = ffn_input[0i] * 2.0;
    up[1i] = ffn_input[1i] * 2.0;
    up[2i] = ffn_input[2i] * 2.0;
    up[3i] = ffn_input[3i] * 2.0;

    var gate_silu = array<f32>([4]);
    gate_silu[0i] = 0.0;
    gate_silu[1i] = 0.0;
    gate_silu[2i] = 0.0;
    gate_silu[3i] = 0.0;

    var gate_activated = silu(gate, gate_silu, dim);

    var ffn_out = array<f32>([4]);
    ffn_out[0i] = gate_activated[0i] * up[0i];
    ffn_out[1i] = gate_activated[1i] * up[1i];
    ffn_out[2i] = gate_activated[2i] * up[2i];
    ffn_out[3i] = gate_activated[3i] * up[3i];

    // 7. Final residual connection
    var final_out = array<f32>([4]);
    final_out[0i] = post_attn[0i] + ffn_out[0i];
    final_out[1i] = post_attn[1i] + ffn_out[1i];
    final_out[2i] = post_attn[2i] + ffn_out[2i];
    final_out[3i] = post_attn[3i] + ffn_out[3i];

    return final_out[0i];
}
