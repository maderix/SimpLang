// Test matrix multiplication with MLIR Linalg backend
// This test performs a simple 2x2 matrix multiplication

fn main() -> f32 {
    // Create matrices:
    // A = [[1.0, 2.0],     (2x2 matrix, stored as 4-element 1D array)
    //      [3.0, 4.0]]
    //
    // B = [[5.0, 6.0],     (2x2 matrix, stored as 4-element 1D array)
    //      [7.0, 8.0]]
    //
    // Expected result C = A × B:
    // C = [[1*5 + 2*7, 1*6 + 2*8],  = [[19.0, 22.0],
    //      [3*5 + 4*7, 3*6 + 4*8]]     [43.0, 50.0]]

    // Create matrix A (2x2, row-major)
    var A = array<f32>([1.0, 2.0, 3.0, 4.0]);

    // Create matrix B (2x2, row-major)
    var B = array<f32>([5.0, 6.0, 7.0, 8.0]);

    // Perform matrix multiplication: C = matmul(A, B, m=2, k=2, n=2)
    // A is 2x2, B is 2x2, so result C will be 2x2
    var C = matmul(A, B, 2, 2, 2);

    // Access result elements:
    // C[0,0] should be 19.0 (1*5 + 2*7)
    // C[0,1] should be 22.0 (1*6 + 2*8)
    // C[1,0] should be 43.0 (3*5 + 4*7)
    // C[1,1] should be 50.0 (3*6 + 4*8)

    // Return C[0,0] to verify correctness (should be 19.0)
    return C[0];
}
