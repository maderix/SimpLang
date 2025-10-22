// Pure matmul kernel - takes pre-initialized arrays
// C++ will handle initialization and benchmarking

fn matmul_kernel(var A array<f32>, var B array<f32>, var m i64, var k i64, var n i64) -> array<f32> {
    return matmul(A, B, m, k, n);
}
