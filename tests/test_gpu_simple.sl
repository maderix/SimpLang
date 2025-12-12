// Simple GPU test - small matmul
fn kernel_main() -> f32 {
    f32<4, 4> A;
    f32<4, 4> B;

    // Initialize A and B
    var i = 0.0;
    while (i < 4.0) {
        var j = 0.0;
        while (j < 4.0) {
            A[i as i64, j as i64] = 1.0;
            B[i as i64, j as i64] = 2.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    // Matmul: C = A * B
    var C = tensor_matmul(A, B);

    // Return C[0,0] which should be 4*2 = 8.0
    return C[0i, 0i];
}
