@rewrite("tile(16,16,16)")
@memory("align(64)")
fn kernel_main() -> f32 {
    f32<64, 64> A;
    f32<64, 64> B;
    f32<64, 64> C;

    for (var i = 0; i < 64; i = i + 1) {
        for (var j = 0; j < 64; j = j + 1) {
            A[i as i64, j as i64] = 0.0;
            B[i as i64, j as i64] = 0.0;
            C[i as i64, j as i64] = 0.0;
        }
    }

    for (var i = 0; i < 64; i = i + 1) {
        for (var j = 0; j < 64; j = j + 1) {
            var sum = 0.0;
            for (var k = 0; k < 64; k = k + 1) {
                sum = sum + A[i as i64, k as i64] * B[k as i64, j as i64];
            }
            C[i as i64, j as i64] = sum;
        }
    }

    return C[0 as i64, 0 as i64];
}
