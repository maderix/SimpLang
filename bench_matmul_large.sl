// Kernel function - receives pre-allocated buffers from host
// Host-kernel model: host allocates, kernel computes
fn matmul_benchmark(f32[] A, f32[] B, f32[] C, i64 m, i64 k, i64 n, i64 iters) -> f32 {
    // Run iterations of matmul (C is reused, accumulates)
    var iter = 0;
    while (iter < iters) {
        matmul(A, B, C, m, k, n);
        iter = iter + 1;
    }

    // Return first element for verification
    return C[0];
}
