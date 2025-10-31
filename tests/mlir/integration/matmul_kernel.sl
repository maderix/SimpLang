// Pure matmul kernel - takes pre-initialized arrays
// C++ will handle initialization and benchmarking

fn matmul_kernel(f32[] A, f32[] B, f32[] C, i64 m, i64 k, i64 n) -> f32 {
    C = matmul(A, B, C, m, k, n, 0i, 0i, 0i);
    return 0.0;
}
