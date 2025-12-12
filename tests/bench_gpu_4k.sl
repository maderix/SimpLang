// GPU f32 MatMul Benchmark - 4K size only
// Uses tensor_fill for fast initialization + tensor_matmul for cuBLAS

fn bench_4096() -> f32 {
    f32<4096, 4096> A;
    f32<4096, 4096> B;

    tensor_fill(A, 1.0);
    tensor_fill(B, 2.0);

    var C = tensor_matmul(A, B);
    return C[0i, 0i];
}

fn kernel_main() -> f32 {
    return bench_4096();
}
