fn kernel_main(var out_sse SSESlice, var out_avx AVXSlice) {
    // Initialize with some values using integer index
    var i = 0i;  // Integer index
    var limit = 2i;  // Adjusted to match the number of vectors
    while (i < limit) {  // Integer comparison
        // SSE Operations - match SIMD version exactly
        slice_set_sse(out_sse, i, sse(1.0, 2.0) * sse(3.0, 4.0));
        
        // AVX Operations - match SIMD version exactly
        slice_set_avx(out_avx, i, avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) * 
                               avx(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
        
        i = i + 1i;  // Integer increment
    }
    
    return 1.0;
} 