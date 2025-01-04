fn kernel_main(var out_sse SSESlice, var out_avx AVXSlice) {
    // ================== SSE Tests ==================
    var sse1 = make(SSESlice, 4);
    var sse2 = make(SSESlice, 4);
    
    // Initialize SSE test vectors
    sse1[0i] = sse(1.0, 2.0);
    sse2[0i] = sse(5.0, 6.0);
    
    // SSE Addition test
    var sse_vec1 = slice_get_sse(sse1, 0i);
    var sse_vec2 = slice_get_sse(sse2, 0i);
    var sse_sum = simd_add(sse_vec1, sse_vec2);
    slice_set_sse(out_sse, 0i, sse_sum);
    
    // SSE Multiplication test
    var sse_prod = simd_mul(sse_vec1, sse_vec2);
    slice_set_sse(out_sse, 1i, sse_prod);
    
    // ================== AVX Tests ==================
    var avx1 = make(AVXSlice, 4);
    var avx2 = make(AVXSlice, 4);
    
    // Initialize AVX test vectors
    avx1[0i] = avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    avx2[0i] = avx(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    
    // AVX Addition test
    var avx_vec1 = slice_get_avx(avx1, 0i);
    var avx_vec2 = slice_get_avx(avx2, 0i);
    var avx_sum = simd_add_avx(avx_vec1, avx_vec2);
    slice_set_avx(out_avx, 0i, avx_sum);
    
    // AVX Multiplication test
    var avx_prod = simd_mul_avx(avx_vec1, avx_vec2);
    slice_set_avx(out_avx, 1i, avx_prod);
    
    // Store reference results for comparison
    slice_set_sse(out_sse, 2i, sse_sum);   // Reference SSE addition result
    slice_set_sse(out_sse, 3i, sse_prod);  // Reference SSE multiplication result
    slice_set_avx(out_avx, 2i, avx_sum);   // Reference AVX addition result
    slice_set_avx(out_avx, 3i, avx_prod);  // Reference AVX multiplication result
}