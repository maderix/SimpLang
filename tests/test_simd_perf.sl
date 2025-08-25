fn kernel_main() -> f32 {
    // Create optimized arrays for heavy computation
    var size = 1024;
    var simd_a = array<f32>([size]);
    var simd_b = array<f32>([size]);
    var simd_c = array<f32>([size]);
    var simd_result = array<f32>([size]);
    
    // Initialize arrays with different patterns
    var i = 0;
    while (i < size) {
        simd_a[i] = i * 0.1 + 1.0;
        simd_b[i] = (i % 100) * 0.05 + 2.0;
        simd_c[i] = (i % 50) * 0.02 + 0.5;
        i = i + 1;
    }
    
    // Heavy computation loop with multiple FMA opportunities
    i = 0;
    while (i < size) {
        var a = simd_a[i];
        var b = simd_b[i];
        var c = simd_c[i];
        
        // Complex polynomial evaluation (benefits from FMA)
        var temp = a * a * b + a * c * c;  // a²b + ac²
        temp = temp + b * b * c;           // + b²c
        temp = temp * a + b * c;           // multiply by a, add bc
        temp = temp * temp;                // square the result
        
        simd_result[i] = temp;
        i = i + 1;
    }
    
    // Second pass: Stencil operation (benefits from aligned loads)
    i = 1;
    while (i < size - 1) {
        var left = simd_result[i - 1];
        var center = simd_result[i];
        var right = simd_result[i + 1];
        
        // Weighted average then square
        simd_result[i] = (center * 0.5 + left * 0.25 + right * 0.25) * (center * 0.5 + left * 0.25 + right * 0.25);
        
        i = i + 1;
    }
    
    // Final reduction
    var sum = 0.0;
    i = 0;
    while (i < size) {
        sum = sum + simd_result[i];
        i = i + 1;
    }
    
    return sum;
}