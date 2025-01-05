fn kernel_main(var out_sse SSESlice, var out_avx AVXSlice) {
    // 1. SSE Tests (128-bit, 2 doubles)
    var sse1 = make(SSESlice, 1);
    var sse2 = make(SSESlice, 1);
    
    // Initialize with easily recognizable patterns
    sse1[0i] = sse(1.0, 2.0);  // [1.0, 2.0]
    sse2[0i] = sse(3.0, 4.0);  // [3.0, 4.0]
    
    // Expected results:
    // ADD: [4.0, 6.0]    (1+3, 2+4)
    // SUB: [-2.0, -2.0]  (1-3, 2-4)
    // MUL: [3.0, 8.0]    (1*3, 2*4)
    // DIV: [0.333, 0.5]  (1/3, 2/4)
    
    slice_set_sse(out_sse, 0i, slice_get_sse(sse1, 0i) + slice_get_sse(sse2, 0i));
    slice_set_sse(out_sse, 1i, slice_get_sse(sse1, 0i) - slice_get_sse(sse2, 0i));
    slice_set_sse(out_sse, 2i, slice_get_sse(sse1, 0i) * slice_get_sse(sse2, 0i));
    slice_set_sse(out_sse, 3i, slice_get_sse(sse1, 0i) / slice_get_sse(sse2, 0i));
    
    // 2. AVX Tests (512-bit, 8 doubles)
    var avx1 = make(AVXSlice, 1);
    var avx2 = make(AVXSlice, 1);
    
    // Initialize with counting patterns - use all 8 elements for AVX
    avx1[0i] = avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);  // Full 8-element initialization
    avx2[0i] = avx(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);  // Full 8-element initialization
    
    // Expected results:
    // ADD: [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]  (each + 2)
    // SUB: [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]  (each - 2)
    // MUL: [2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0]  (each * 2)
    // DIV: [0.5, 0.67, 0.75, 0.8, 0.83, 0.86, 0.88, 0.89]  (each / 2)
    
    slice_set_avx(out_avx, 0i, slice_get_avx(avx1, 0i) + slice_get_avx(avx2, 0i));
    slice_set_avx(out_avx, 1i, slice_get_avx(avx1, 0i) - slice_get_avx(avx2, 0i));
    slice_set_avx(out_avx, 2i, slice_get_avx(avx1, 0i) * slice_get_avx(avx2, 0i));
    slice_set_avx(out_avx, 3i, slice_get_avx(avx1, 0i) / slice_get_avx(avx2, 0i));
    
    return 1.0;
}