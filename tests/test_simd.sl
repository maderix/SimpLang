fn kernel_main(var out_sse SSESlice, var out_avx AVXSlice) {
    // SSE computations
    var sse1 = make(SSESlice, 4);
    sse1[0] = sse(1.0, 2.0, 3.0, 4.0);
    
    var sse2 = make(SSESlice, 4);
    sse2[0] = sse(5.0, 6.0, 7.0, 8.0);
    
    var sse_sum = simd_add(sse1[0], sse2[0]);
    out_sse[0] = sse_sum;
    
    var sse3 = make(SSESlice, 4);
    sse3[0] = sse(2.0, 3.0, 4.0, 5.0);
    
    var sse4 = make(SSESlice, 4);
    sse4[0] = sse(3.0, 4.0, 5.0, 6.0);
    
    var sse_prod = simd_mul(sse3[0], sse4[0]);
    out_sse[1] = sse_prod;
    
    // AVX computations
    var avx1 = make(AVXSlice, 4);
    avx1[0] = avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    
    var avx2 = make(AVXSlice, 4);
    avx2[0] = avx(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    
    var avx_sum = simd_add(avx1[0], avx2[0]);
    out_avx[0] = avx_sum;
    
    var avx3 = make(AVXSlice, 4);
    avx3[0] = avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    
    var avx4 = make(AVXSlice, 4);
    avx4[0] = avx(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    
    var avx_prod = simd_mul(avx3[0], avx4[0]);
    out_avx[1] = avx_prod;
}