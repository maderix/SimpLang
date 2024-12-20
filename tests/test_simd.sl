fn kernel_main(var out_sse SSESlice, var out_avx AVXSlice) {
    // 2. AVX Test
    var avx1 = make(AVXSlice, 2);
    
    // Test single AVX vector operation - explicitly pass all 8 values
    avx1[0i] = avx(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0);  // Explicit 8 values
    avx1[1i] = avx(5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0);  // Explicit 8 values
    
    // Store result
    slice_set_avx(out_avx, 0i, slice_get_avx(avx1, 0i));
    slice_set_avx(out_avx, 1i, slice_get_avx(avx1, 1i));
    
    return 1.0;
}