// Simple attention test: QK^T * V pattern
// For now, testing the basic operations without full attention
fn test_attention_simple() -> f32 {
    var seq_len = 2i;
    var d_k = 2i;

    // Query: [seq_len, d_k]
    var Q = array<f32>([4]);
    Q[0i] = 1.0;  // Q[0,0]
    Q[1i] = 0.0;  // Q[0,1]
    Q[2i] = 0.0;  // Q[1,0]
    Q[3i] = 1.0;  // Q[1,1]

    // Key: [seq_len, d_k]
    var K = array<f32>([4]);
    K[0i] = 1.0;  // K[0,0]
    K[1i] = 0.0;  // K[0,1]
    K[2i] = 0.0;  // K[1,0]
    K[3i] = 1.0;  // K[1,1]

    // Value: [seq_len, d_k]
    var V = array<f32>([4]);
    V[0i] = 2.0;  // V[0,0]
    V[1i] = 0.0;  // V[0,1]
    V[2i] = 0.0;  // V[1,0]
    V[3i] = 3.0;  // V[1,1]

    // Scores: Q * K^T = [seq_len, seq_len]
    var scores = array<f32>([4]);
    scores[0i] = 0.0;
    scores[1i] = 0.0;
    scores[2i] = 0.0;
    scores[3i] = 0.0;

    // Manual QK^T computation for [2x2] @ [2x2]^T
    // scores[0,0] = Q[0,:] * K[0,:]
    scores[0i] = Q[0i] * K[0i] + Q[1i] * K[1i];  // 1*1 + 0*0 = 1
    scores[1i] = Q[0i] * K[2i] + Q[1i] * K[3i];  // 1*0 + 0*1 = 0
    scores[2i] = Q[2i] * K[0i] + Q[3i] * K[1i];  // 0*1 + 1*0 = 0
    scores[3i] = Q[2i] * K[2i] + Q[3i] * K[3i];  // 0*0 + 1*1 = 1

    // Apply softmax to first row of scores
    var row1 = array<f32>([2]);
    row1[0i] = scores[0i];
    row1[1i] = scores[1i];

    var row1_softmax = array<f32>([2]);
    row1_softmax[0i] = 0.0;
    row1_softmax[1i] = 0.0;

    var row1_result = softmax(row1, row1_softmax, 2i);

    // Attention output: softmax(scores) * V
    // For first position: row1_result[0] * V[0,:] + row1_result[1] * V[1,:]
    var out0 = row1_result[0i] * V[0i] + row1_result[1i] * V[2i];

    return out0;
}
