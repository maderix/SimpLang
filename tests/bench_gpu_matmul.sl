// GPU MatMul Benchmark - 1024x1024
fn kernel_main() -> f32 {
    f32<1024, 1024> A;
    f32<1024, 1024> B;
    
    // Initialize using index variables
    for (var i in 0:1024) {
        for (var j in 0:1024) {
            A[i, j] = 1.0;
            B[i, j] = 2.0;
        }
    }
    
    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}
