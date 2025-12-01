// Test matrix multiplication with annotations for optimization

@rewrite("tile(16,16,16)")
@memory("align(64)")
fn tensor_matmul_256(f32<256, 256> A, f32<256, 256> B) -> f32<256, 256> {
    f32<256, 256> C;

    // Initialize result matrix to zero
    for (var i = 0.0; i < 256.0; i = i + 1.0) {
        for (var j = 0.0; j < 256.0; j = j + 1.0) {
            C[i as i64, j as i64] = 0.0;
        }
    }

    // Matrix multiplication: C = A * B
    for (var i = 0.0; i < 256.0; i = i + 1.0) {
        for (var j = 0.0; j < 256.0; j = j + 1.0) {
            var sum = 0.0;
            for (var k = 0.0; k < 256.0; k = k + 1.0) {
                sum = sum + A[i as i64, k as i64] * B[k as i64, j as i64];
            }
            C[i as i64, j as i64] = sum;
        }
    }

    return C;
}

fn kernel_main() -> f32 {
    // Initialize 256x256 matrices
    f32<256, 256> A;
    f32<256, 256> B;

    var i = 0.0;
    for (i = 0.0; i < 256.0; i = i + 1.0) {
        var j = 0.0;
        for (j = 0.0; j < 256.0; j = j + 1.0) {
            var idx_i = i as i64;
            var idx_j = j as i64;
            A[idx_i, idx_j] = (i * 256.0 + j) / 65536.0;
            B[idx_i, idx_j] = (j * 256.0 + i) / 65536.0;
        }
    }

    // Call our annotated matmul
    var C = tensor_matmul_256(A, B);

    // Compute checksum
    var checksum = 0.0;
    for (i = 0.0; i < 256.0; i = i + 1.0) {
        for (var j = 0.0; j < 256.0; j = j + 1.0) {
            checksum = checksum + C[i as i64, j as i64];
        }
    }

    return checksum;
}
