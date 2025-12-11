// GPU f32 MatMul Benchmark
// Uses tensor_matmul which triggers cuBLAS on GPU

fn bench_256() -> f32 {
    f32<256, 256> A;
    f32<256, 256> B;

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

    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}

fn bench_512() -> f32 {
    f32<512, 512> A;
    f32<512, 512> B;

    var i = 0.0;
    while (i < 512.0) {
        var j = 0.0;
        while (j < 512.0) {
            A[i as i64, j as i64] = 1.0;
            B[i as i64, j as i64] = 2.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}

fn bench_1024() -> f32 {
    f32<1024, 1024> A;
    f32<1024, 1024> B;

    var i = 0.0;
    while (i < 1024.0) {
        var j = 0.0;
        while (j < 1024.0) {
            A[i as i64, j as i64] = 1.0;
            B[i as i64, j as i64] = 2.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}

fn bench_2048() -> f32 {
    f32<2048, 2048> A;
    f32<2048, 2048> B;

    var i = 0.0;
    while (i < 2048.0) {
        var j = 0.0;
        while (j < 2048.0) {
            A[i as i64, j as i64] = 1.0;
            B[i as i64, j as i64] = 2.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}

fn kernel_main() -> f32 {
    return bench_256();
}
