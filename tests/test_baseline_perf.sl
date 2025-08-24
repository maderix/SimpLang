fn kernel_main() {
    // Create regular arrays (no SIMD alignment/hints)
    var size = 2097152;  // 1M elements to showcase SIMD benefits
    var regular_a = array<f32>([size]);
    var regular_b = array<f32>([size]);
    var regular_c = array<f32>([size]);
    var regular_result = array<f32>([size]);
    
    // Initialize arrays with same patterns as SIMD version
    var i = 0;
    while (i < size) {
        regular_a[i] = i * 0.1 + 1.0;
        regular_b[i] = (i % 100) * 0.05 + 2.0;
        regular_c[i] = (i % 50) * 0.02 + 0.5;
        i = i + 1;
    }
    
    // Same heavy computation as SIMD version
    i = 0;
    while (i < size) {
        var a = regular_a[i];
        var b = regular_b[i];
        var c = regular_c[i];
        
        // Complex polynomial evaluation (same as SIMD)
        var temp = a * a * b + a * c * c;  // a²b + ac²
        temp = temp + b * b * c;           // + b²c
        temp = temp * a + b * c;           // multiply by a, add bc
        temp = temp * temp;                // square the result
        
        regular_result[i] = temp;
        i = i + 1;
    }
    
    // Second pass: Same stencil operation
    i = 1;
    while (i < size - 1) {
        var left = regular_result[i - 1];
        var center = regular_result[i];
        var right = regular_result[i + 1];
        
        // Weighted average then square
        regular_result[i] = (center * 0.5 + left * 0.25 + right * 0.25) * (center * 0.5 + left * 0.25 + right * 0.25);
        
        i = i + 1;
    }
    
    // Final reduction
    var sum = 0.0;
    i = 0;
    while (i < size) {
        sum = sum + regular_result[i];
        i = i + 1;
    }
    
    return sum;
}