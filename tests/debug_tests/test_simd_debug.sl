fn test_sse_ops() {
    var vec1 = make(SSESlice, 4);
    var vec2 = make(SSESlice, 4);
    var result = make(SSESlice, 4);

    // Initialize vectors
    vec1[0i] = sse(1.0, 2.0, 3.0, 4.0);
    vec2[0i] = sse(5.0, 6.0, 7.0, 8.0);

    // Test addition
    var a = slice_get_sse(vec1, 0i);
    var b = slice_get_sse(vec2, 0i);
    var sum = simd_add(a, b);
    slice_set_sse(result, 0i, sum);

    // Test multiplication
    var prod = simd_mul(a, b);
    slice_set_sse(result, 1i, prod);

    return 1.0;  // Return success
}

fn test_avx_ops() {
    var vec1 = make(AVXSlice, 4);
    var vec2 = make(AVXSlice, 4);
    var result = make(AVXSlice, 4);

    // Initialize vectors
    vec1[0i] = avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    vec2[0i] = avx(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

    // Test addition
    var a = slice_get_avx(vec1, 0i);
    var b = slice_get_avx(vec2, 0i);
    var sum = simd_add_avx(a, b);
    slice_set_avx(result, 0i, sum);

    // Test multiplication
    var prod = simd_mul_avx(a, b);
    slice_set_avx(result, 1i, prod);

    return 1.0;  // Return success
}

fn kernel_main() {
    var sse_result = test_sse_ops();
    var avx_result = test_avx_ops();
    return sse_result + avx_result;  // Return combined success value
}