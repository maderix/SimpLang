// SimpLang Tensor Matrix Operations Test
// Test 2D tensor operations like matrix multiplication and transpose

func kernel_main() -> f32 {
    // Test small matrix operations for correctness
    var M = 64;   // Rows of A, rows of C
    var K = 64;   // Cols of A, rows of B  
    var N = 64;   // Cols of B, cols of C
    
    // Create matrices A[M x K], B[K x N], C[M x N]
    var A = tensor_create_f32(M * K);
    var B = tensor_create_f32(K * N);  
    var C = tensor_create_f32(M * N);
    
    // Initialize test matrices
    // A = identity-like pattern, B = constant values
    var i = 0;
    while (i < M) {
        var j = 0;
        while (j < K) {
            var idx = tensor_index_2d(i, j, K);
            if (i == j) {
                A[idx] = 1.0;  // Diagonal elements
            } else {
                A[idx] = 0.1;  // Off-diagonal elements
            }
            j = j + 1;
        }
        i = i + 1;
    }
    
    // Fill B with test values
    tensor_fill_f32(B, 2.0, K * N);
    
    // Test matrix multiplication: C = A * B
    tensor_matmul_f32(A, B, C, M, K, N);
    
    // Test matrix transpose
    var A_T = tensor_create_f32(K * M);  // A^T is [K x M]
    tensor_transpose_f32(A, A_T, M, K);
    
    // Verify results by summing C matrix
    var result_sum = tensor_sum_f32(C, M * N);
    
    // Test larger matrix for performance (should trigger SIMD)
    var large_M = 256;
    var large_N = 256;
    var large_K = 256;
    
    var A_large = tensor_create_f32(large_M * large_K);
    var B_large = tensor_create_f32(large_K * large_N);
    var C_large = tensor_create_f32(large_M * large_N);
    
    // Initialize with simple patterns for performance test
    tensor_fill_f32(A_large, 1.5, large_M * large_K);
    tensor_fill_f32(B_large, 2.0, large_K * large_N);
    
    // Large matrix multiply - should show SIMD benefits
    tensor_matmul_f32(A_large, B_large, C_large, large_M, large_K, large_N);
    
    var large_sum = tensor_sum_f32(C_large, large_M * large_N);
    
    return large_sum;
}