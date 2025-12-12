// GPU MatMul-Only Benchmark - Host passes all arrays (A, B, C)

fn bench_matmul_2048(f32[] a_arr, f32[] b_arr, f32[] c_arr) -> f32 {
    f32<2048, 2048> A = tensor_from_array(a_arr, 0i);
    f32<2048, 2048> B = tensor_from_array(b_arr, 0i);
    f32<2048, 2048> C = tensor_from_array(c_arr, 0i);
    tensor_matmul_into(A, B, C);
    return C[0i, 0i];
}

fn bench_matmul_4096(f32[] a_arr, f32[] b_arr, f32[] c_arr) -> f32 {
    f32<4096, 4096> A = tensor_from_array(a_arr, 0i);
    f32<4096, 4096> B = tensor_from_array(b_arr, 0i);
    f32<4096, 4096> C = tensor_from_array(c_arr, 0i);
    tensor_matmul_into(A, B, C);
    return C[0i, 0i];
}

fn bench_matmul_8192(f32[] a_arr, f32[] b_arr, f32[] c_arr) -> f32 {
    f32<8192, 8192> A = tensor_from_array(a_arr, 0i);
    f32<8192, 8192> B = tensor_from_array(b_arr, 0i);
    f32<8192, 8192> C = tensor_from_array(c_arr, 0i);
    tensor_matmul_into(A, B, C);
    return C[0i, 0i];
}

fn kernel_main() -> f32 {
    return 0.0;
}
