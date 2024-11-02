fn kernel_main(var out_sse SSESlice, var out_avx AVXSlice) {
    // ================== Input Arrays ==================
    
    // SSE Input Arrays (4-wide)
    // Addition inputs:
    //   Array1: [1.0, 2.0, 3.0, 4.0]
    //   Array2: [5.0, 6.0, 7.0, 8.0]
    //
    // Multiplication inputs:
    //   Array1: [2.0, 3.0, 4.0, 5.0]
    //   Array2: [3.0, 4.0, 5.0, 6.0]
    
    // AVX Input Arrays (8-wide)
    // Addition inputs:
    //   Array1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    //   Array2: [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    //
    // Multiplication inputs:
    //   Array1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    //   Array2: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    // ================== Reference Results ==================
    
    // 1. SSE Reference Results (4-wide)
    var ref_sse = make(SSESlice, 4);
    
    // Addition: [1.0, 2.0, 3.0, 4.0] + [5.0, 6.0, 7.0, 8.0]
    // Expected: [6.0, 8.0, 10.0, 12.0]
    ref_sse[0i] = sse(1.0 + 5.0,  // 6.0
                      2.0 + 6.0,  // 8.0
                      3.0 + 7.0,  // 10.0
                      4.0 + 8.0); // 12.0
    
    // Multiplication: [2.0, 3.0, 4.0, 5.0] * [3.0, 4.0, 5.0, 6.0]
    // Expected: [6.0, 12.0, 20.0, 30.0]
    ref_sse[1i] = sse(2.0 * 3.0,  // 6.0
                      3.0 * 4.0,  // 12.0
                      4.0 * 5.0,  // 20.0
                      5.0 * 6.0); // 30.0

    // 2. AVX Reference Results (8-wide)
    var ref_avx = make(AVXSlice, 4);
    
    // Addition: [1,2,3,4,5,6,7,8] + [8,7,6,5,4,3,2,1]
    // Expected: [9,9,9,9,9,9,9,9]
    ref_avx[0i] = avx(1.0 + 8.0,  // 9.0
                      2.0 + 7.0,  // 9.0
                      3.0 + 6.0,  // 9.0
                      4.0 + 5.0,  // 9.0
                      5.0 + 4.0,  // 9.0
                      6.0 + 3.0,  // 9.0
                      7.0 + 2.0,  // 9.0
                      8.0 + 1.0); // 9.0
    
    // Multiplication: [1,2,3,4,5,6,7,8] * [2,3,4,5,6,7,8,9]
    // Expected: [2,6,12,20,30,42,56,72]
    ref_avx[1i] = avx(1.0 * 2.0,  // 2.0
                      2.0 * 3.0,  // 6.0
                      3.0 * 4.0,  // 12.0
                      4.0 * 5.0,  // 20.0
                      5.0 * 6.0,  // 30.0
                      6.0 * 7.0,  // 42.0
                      7.0 * 8.0,  // 56.0
                      8.0 * 9.0); // 72.0

    // ================== SIMD Operations ==================
    
    // 1. SSE SIMD Operations
    var sse1 = make(SSESlice, 4);
    sse1[0i] = sse(1.0, 2.0, 3.0, 4.0);
    var sse2 = make(SSESlice, 4);
    sse2[0i] = sse(5.0, 6.0, 7.0, 8.0);
    
    var sse_vec1 = slice_get_sse(sse1, 0i);
    var sse_vec2 = slice_get_sse(sse2, 0i);
    var sse_sum = simd_add(sse_vec1, sse_vec2);
    slice_set_sse(out_sse, 0i, sse_sum);
    
    var sse3 = make(SSESlice, 4);
    sse3[0i] = sse(2.0, 3.0, 4.0, 5.0);
    var sse4 = make(SSESlice, 4);
    sse4[0i] = sse(3.0, 4.0, 5.0, 6.0);
    
    var sse_vec3 = slice_get_sse(sse3, 0i);
    var sse_vec4 = slice_get_sse(sse4, 0i);
    var sse_prod = simd_mul(sse_vec3, sse_vec4);
    slice_set_sse(out_sse, 1i, sse_prod);
    
    // 2. AVX SIMD Operations
    var avx1 = make(AVXSlice, 4);
    avx1[0i] = avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    var avx2 = make(AVXSlice, 4);
    avx2[0i] = avx(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    
    var avx_vec1 = slice_get_avx(avx1, 0i);
    var avx_vec2 = slice_get_avx(avx2, 0i);
    var avx_sum = simd_add_avx(avx_vec1, avx_vec2);
    slice_set_avx(out_avx, 0i, avx_sum);
    
    var avx3 = make(AVXSlice, 4);
    avx3[0i] = avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    var avx4 = make(AVXSlice, 4);
    avx4[0i] = avx(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    
    var avx_vec3 = slice_get_avx(avx3, 0i);
    var avx_vec4 = slice_get_avx(avx4, 0i);
    var avx_prod = simd_mul_avx(avx_vec3, avx_vec4);
    slice_set_avx(out_avx, 1i, avx_prod);
    
    // Store results for comparison
    slice_set_sse(out_sse, 2i, ref_sse[0i]);
    slice_set_sse(out_sse, 3i, ref_sse[1i]);
    slice_set_avx(out_avx, 2i, ref_avx[0i]);
    slice_set_avx(out_avx, 3i, ref_avx[1i]);
    
    return 1.0;  // Return success value
}