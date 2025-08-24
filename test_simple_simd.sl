fn kernel_main() {
    // Simple test: try to create an AVX-512 array
    var simd_arr = array<f32, simd=avx512>([4]);
    simd_arr[0] = 1.0;
    simd_arr[1] = 2.0;
    return simd_arr[0] + simd_arr[1];  // Should return 3.0
}