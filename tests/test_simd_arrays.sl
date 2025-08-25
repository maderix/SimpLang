fn kernel_main() {
    // Test 1: Regular array (baseline)
    var regular_arr = array<f32>([16]);
    var i = 0;
    while (i < 16) {
        regular_arr[i] = i * 2.0;
        i = i + 1;
    }
    
    // Test 2: AVX-512 SIMD array
    var simd_arr = array<f32, simd=avx512>([16]);
    i = 0;
    while (i < 16) {
        simd_arr[i] = i * 3.0;
        i = i + 1;
    }
    
    // Test 3: Auto SIMD array (should pick best available)
    var auto_simd = array<f32, simd=auto>([16]);
    i = 0;
    while (i < 16) {
        auto_simd[i] = i * 1.5;
        i = i + 1;
    }
    
    // Compute results for verification
    var regular_sum = 0.0;
    var simd_sum = 0.0;
    var auto_sum = 0.0;
    
    i = 0;
    while (i < 16) {
        regular_sum = regular_sum + regular_arr[i];
        simd_sum = simd_sum + simd_arr[i];
        auto_sum = auto_sum + auto_simd[i];
        i = i + 1;
    }
    
    // Expected: regular=240.0 (0+2+4+...+30), simd=360.0, auto=180.0
    return regular_sum + simd_sum + auto_sum;  // Should be 780.0
}