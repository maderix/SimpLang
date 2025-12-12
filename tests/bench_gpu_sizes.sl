// GPU f32 MatMul Benchmark - Multiple sizes
// Uses tensor_fill + tensor_matmul for cuBLAS

fn bench_1024() -> f32 {
    f32<1024, 1024> A;
    f32<1024, 1024> B;
    tensor_fill(A, 1.0);
    tensor_fill(B, 2.0);
    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}

fn bench_2048() -> f32 {
    f32<2048, 2048> A;
    f32<2048, 2048> B;
    tensor_fill(A, 1.0);
    tensor_fill(B, 2.0);
    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}

fn bench_4096() -> f32 {
    f32<4096, 4096> A;
    f32<4096, 4096> B;
    tensor_fill(A, 1.0);
    tensor_fill(B, 2.0);
    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}

fn bench_6144() -> f32 {
    f32<6144, 6144> A;
    f32<6144, 6144> B;
    tensor_fill(A, 1.0);
    tensor_fill(B, 2.0);
    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}

fn bench_8192() -> f32 {
    f32<8192, 8192> A;
    f32<8192, 8192> B;
    tensor_fill(A, 1.0);
    tensor_fill(B, 2.0);
    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}

fn kernel_main() -> f32 {
    return bench_1024();
}
