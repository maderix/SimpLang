// Performance test kernel for SIMD operations
// Corrected version: ensures 'size', 'i', and 'limit' are i64

// You might have a syntax to explicitly say "var size i64 = 4;" or "var size = 4i;" 
// depending on your compiler/language rules. 
// The key is: your AST/codegen must actually treat these as i64, not double.

fn kernel_main(var out_sse SSESlice, var out_avx AVXSlice) {
    // Use an integer (i64) size
    var size = 4i;  // or "var size = 4i;" if that truly yields i64
    var sse1 = make(SSESlice, size);
    var sse2 = make(SSESlice, size);
    var avx1 = make(AVXSlice, size);
    var avx2 = make(AVXSlice, size);
    
    // Integer counters
    var i = 0i;
    var limit = 4i;

    // 1) Initialize slices
    while (i < limit) {
        slice_set_sse(sse1, i, sse(1.0, 2.0));
        slice_set_sse(sse2, i, sse(3.0, 4.0));
        
        slice_set_avx(avx1, i, avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0));
        slice_set_avx(avx2, i, avx(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
        
        i = i + 1;  // increment as i64
    }
    
    // 2) Perform “complex” computation
    i = 0;  // reset
    while (i < limit) {
        // SSE multiply
        slice_set_sse(
            out_sse, 
            i, 
            slice_get_sse(sse1, i) * slice_get_sse(sse2, i)
        );
        // AVX multiply
        slice_set_avx(
            out_avx, 
            i, 
            slice_get_avx(avx1, i) * slice_get_avx(avx2, i)
        );
        
        i = i + 1i; // increment as i64
    }
    
    return 1.0;
}
