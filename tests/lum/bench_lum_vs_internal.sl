// Benchmark: Lum DSL vs SimpLang Internal Tiling
// This benchmark compares performance of:
// 1. SimpLang internal tiling (hierarchical 32x32 outer, 16x16 inner)
// 2. Lum-controlled tiling via Transform Dialect

fn matmul_512() -> f32 {
    f32<512, 512> A;
    f32<512, 512> B;

    // Initialize matrices
    var i = 0.0;
    while (i < 512.0) {
        var j = 0.0;
        while (j < 512.0) {
            A[i as i64, j as i64] = (i * 512.0 + j) / 262144.0;
            B[i as i64, j as i64] = (j * 512.0 + i) / 262144.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    // Matmul - will be tiled by either SimpLang or Lum
    var C = tensor_matmul(A, B);

    // Compute checksum for verification
    var checksum = 0.0;
    i = 0.0;
    while (i < 512.0) {
        var j = 0.0;
        while (j < 512.0) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    return checksum;
}
