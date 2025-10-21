// Simple matrix multiplication test with MLIR
// Creates arrays and manually sets values

fn main() -> f32 {
    // Create 2x2 matrix A (4 elements total)
    var A = array<f32>([4]);
    A[0] = 1.0;
    A[1] = 2.0;
    A[2] = 3.0;
    A[3] = 4.0;

    // Create 2x2 matrix B (4 elements total)
    var B = array<f32>([4]);
    B[0] = 5.0;
    B[1] = 6.0;
    B[2] = 7.0;
    B[3] = 8.0;

    // Perform matrix multiplication: C = matmul(A, B, m=2, k=2, n=2)
    // A is 2x2, B is 2x2, result C will be 2x2
    // Expected:
    // C = [[1*5 + 2*7, 1*6 + 2*8],  = [[19, 22],
    //      [3*5 + 4*7, 3*6 + 4*8]]     [43, 50]]
    var C = matmul(A, B, 2, 2, 2);

    // Return C[0] which should be 19.0
    return C[0];
}
