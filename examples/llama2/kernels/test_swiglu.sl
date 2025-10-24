// SwiGLU FFN test
// SwiGLU(x) = SiLU(x * W_gate) ⊙ (x * W_up)
// Where ⊙ is element-wise multiplication

fn test_swiglu() -> f32 {
    var dim = 4i;

    // Input vector
    var x = array<f32>([4]);
    x[0i] = 1.0;
    x[1i] = 2.0;
    x[2i] = 3.0;
    x[3i] = 4.0;

    // For simplicity, using identity-like transformations
    // gate_proj: apply some transformation
    var gate = array<f32>([4]);
    gate[0i] = x[0i] * 0.5;  // Simple scaling
    gate[1i] = x[1i] * 0.5;
    gate[2i] = x[2i] * 0.5;
    gate[3i] = x[3i] * 0.5;

    // up_proj: another transformation
    var up = array<f32>([4]);
    up[0i] = x[0i] * 2.0;
    up[1i] = x[1i] * 2.0;
    up[2i] = x[2i] * 2.0;
    up[3i] = x[3i] * 2.0;

    // Apply SiLU to gate
    var gate_silu = array<f32>([4]);
    gate_silu[0i] = 0.0;
    gate_silu[1i] = 0.0;
    gate_silu[2i] = 0.0;
    gate_silu[3i] = 0.0;

    var gate_activated = silu(gate, gate_silu, dim);

    // Element-wise multiply: gate_activated ⊙ up
    var result = array<f32>([4]);
    result[0i] = gate_activated[0i] * up[0i];
    result[1i] = gate_activated[1i] * up[1i];
    result[2i] = gate_activated[2i] * up[2i];
    result[3i] = gate_activated[3i] * up[3i];

    return result[0i];
}
