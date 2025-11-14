// Simplified matrix multiplication benchmark
// 32x32 matrices, single multiplication

fn main() -> f32 {
    // Create 32x32 matrices (1024 elements each)
    var A = array<f32>([1024]);
    var B = array<f32>([1024]);
    var C = array<f32>([1024]);

    // Initialize first few elements to test
    A[0] = 1.0;
    A[1] = 1.0;
    A[2] = 1.0;
    A[31] = 1.0;
    A[32] = 1.0;
    A[63] = 1.0;

    B[0] = 2.0;
    B[1] = 2.0;
    B[2] = 2.0;
    B[31] = 2.0;
    B[32] = 2.0;
    B[63] = 2.0;

    // Perform matrix multiplication
    // Even with sparse initialization, this computes full 32x32x32 = ~65K FLOPs
    C = matmul(A, B, C, 32, 32, 32, 0, 0, 0);

    // Return first element
    return C[0];
}
