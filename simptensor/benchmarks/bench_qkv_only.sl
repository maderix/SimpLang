// Isolated QKV Projection Benchmark - ALL USING PRE-TRANSPOSED WEIGHTS
// Single token decode: [1, 2048] @ [2048, 2048/512]
// All weights are pre-transposed: Wq_T has shape [N, K] instead of [K, N]

// Single token Q projection M=1
fn bench_q_m1(
    i8[] x,       // [2048] input
    i8[] Wq_T,    // [2048, 2048] weights PRE-TRANSPOSED: shape [N, K]
    i64 dim       // 2048
) -> i32 {
    i8<1, 2048> x_row = tensor_from_array(x, 0i);
    i8<2048, 2048> Wq_T_mat = tensor_from_array(Wq_T, 0i);
    var Q = tensor_matmul_nt(x_row, Wq_T_mat);
    var checksum = Q[0i, 0i] + Q[0i, 1023i] + Q[0i, 2047i];
    return checksum;
}

// Batched Q projection M=4
fn bench_q_m4(
    i8[] X,       // [4, 2048]
    i8[] Wq_T,    // [2048, 2048] PRE-TRANSPOSED
    i64 batch,
    i64 dim
) -> i32 {
    i8<4, 2048> X_mat = tensor_from_array(X, 0i);
    i8<2048, 2048> Wq_T_mat = tensor_from_array(Wq_T, 0i);
    var Q = tensor_matmul_nt(X_mat, Wq_T_mat);
    var checksum = Q[0i, 0i] + Q[1i, 1023i] + Q[3i, 2047i];
    return checksum;
}

// Batched Q projection M=16
fn bench_q_m16(
    i8[] X,       // [16, 2048]
    i8[] Wq_T,    // [2048, 2048] PRE-TRANSPOSED
    i64 batch,
    i64 dim
) -> i32 {
    i8<16, 2048> X_mat = tensor_from_array(X, 0i);
    i8<2048, 2048> Wq_T_mat = tensor_from_array(Wq_T, 0i);
    var Q = tensor_matmul_nt(X_mat, Wq_T_mat);
    var checksum = Q[0i, 0i] + Q[7i, 1023i] + Q[15i, 2047i];
    return checksum;
}

// Batched Q projection M=128 (prefill)
fn bench_q_m128(
    i8[] X,       // [128, 2048]
    i8[] Wq_T,    // [2048, 2048] PRE-TRANSPOSED
    i64 batch,
    i64 dim
) -> i32 {
    i8<128, 2048> X_mat = tensor_from_array(X, 0i);
    i8<2048, 2048> Wq_T_mat = tensor_from_array(Wq_T, 0i);
    var Q = tensor_matmul_nt(X_mat, Wq_T_mat);
    var checksum = Q[0i, 0i] + Q[63i, 1023i] + Q[127i, 2047i];
    return checksum;
}
