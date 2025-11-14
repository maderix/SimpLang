// Matrix multiplication benchmark
// Multiplies two NxN matrices multiple times

fn main() -> f32 {
    // Create 64x64 matrices (4096 elements each)
    var N = 64;
    var size = 4096;  // N * N

    // Matrix A (64x64)
    var A = array<f32>([4096]);
    var i = 0;
    while (i < size) {
        A[i] = 1.0;  // Initialize with 1.0
        i = i + 1;
    }

    // Matrix B (64x64)
    var B = array<f32>([4096]);
    i = 0;
    while (i < size) {
        B[i] = 2.0;  // Initialize with 2.0
        i = i + 1;
    }

    // Perform matrix multiplication 10 times
    var iterations = 10;
    var iter = 0;
    var C = array<f32>([4096]);

    while (iter < iterations) {
        C = matmul(A, B, C, 64, 64, 64, 0, 0, 0);
        iter = iter + 1;
    }

    // Return first element
    // Expected: 64 * 1.0 * 2.0 = 128.0
    return C[0];
}
