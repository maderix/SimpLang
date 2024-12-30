// Performance test kernel for SIMD operations
// Performs complex mathematical operations on large slices

fn kernel_main(var out_sse SSESlice, var out_avx AVXSlice) {
    // Create work slices with integer size - smaller size for testing
    var size = 4i;  // Explicit integer constant
    var sse1 = make(SSESlice, size);
    var sse2 = make(SSESlice, size);
    var avx1 = make(AVXSlice, size);
    var avx2 = make(AVXSlice, size);
    
    // Initialize with some values using integer index
    var i = 0i;  // Integer index
    var limit = 4i;  // Integer limit
    while (i < limit) {  // Integer comparison
        // SSE initialization
        slice_set_sse(sse1, i, sse(1.0, 2.0));
        slice_set_sse(sse2, i, sse(3.0, 4.0));
        
        // AVX initialization
        slice_set_avx(avx1, i, avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0));
        slice_set_avx(avx2, i, avx(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
        
        i = i + 1i;  // Integer increment
    }
    
    // Complex computation loop
    i = 0i;  // Reset integer index
    while (i < limit) {  // Integer comparison
        // SSE Operations
        slice_set_sse(out_sse, i, slice_get_sse(sse1, i) * slice_get_sse(sse2, i));
        
        // AVX Operations
        slice_set_avx(out_avx, i, slice_get_avx(avx1, i) * slice_get_avx(avx2, i));
        
        i = i + 1i;  // Integer increment
    }
    
    return 1.0;
} 