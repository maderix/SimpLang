// Large GPU matmul test - should trigger cuBLAS
fn kernel_main() -> f32 {
    f32<256, 256> A;
    f32<256, 256> B;

    // Initialize with simple values
    var i = 0.0;
    while (i < 256.0) {
        var j = 0.0;
        while (j < 256.0) {
            A[i as i64, j as i64] = 1.0;
            B[i as i64, j as i64] = 2.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    // Matmul: C = A * B (256x256, should use cuBLAS)
    var C = tensor_matmul(A, B);

    // Return C[0,0] which should be 256*1*2 = 512.0
    return C[0i, 0i];
}
