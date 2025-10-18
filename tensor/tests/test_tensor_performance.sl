// SimpLang Tensor Performance Test
// Benchmark tensor operations to verify SIMD auto-vectorization benefits

func kernel_main() -> f32 {
    var large_size = 2097152;  // 2M elements for clear SIMD benefits
    
    // Test F32 performance - should show significant SIMD speedup
    var a = tensor_create_f32(large_size);
    var b = tensor_create_f32(large_size);
    var c = tensor_create_f32(large_size);
    var d = tensor_create_f32(large_size);
    
    // Fill arrays with test data
    tensor_fill_f32(a, 3.14159, large_size);
    tensor_fill_f32(b, 2.71828, large_size);
    
    // Perform multiple operations to stress-test SIMD vectorization
    tensor_add_f32(a, b, c, large_size);       // c = a + b
    tensor_mul_f32(c, a, d, large_size);       // d = c * a
    tensor_sub_f32(d, b, c, large_size);       // c = d - b
    
    // Reduction operations (horizontal SIMD)
    var sum = tensor_sum_f32(c, large_size);
    var mean = tensor_mean_f32(c, large_size);
    var max_val = tensor_max_f32(c, large_size);
    var min_val = tensor_min_f32(c, large_size);
    
    // Dot product (multiply + reduction)
    var dot = tensor_dot_f32(a, b, large_size);
    
    // Complex operations combining multiple SIMD patterns
    var temp1 = tensor_create_f32(large_size);
    var temp2 = tensor_create_f32(large_size);
    
    // Fused multiply-add pattern: temp1 = a * 2.0 + b
    tensor_mul_scalar_f32(a, 2.0, temp1, large_size);
    tensor_add_f32(temp1, b, temp2, large_size);
    
    // Chain operations for pipeline optimization
    tensor_mul_f32(temp2, a, temp1, large_size);
    tensor_add_scalar_f32(temp1, 1.0, temp2, large_size);
    
    var final_sum = tensor_sum_f32(temp2, large_size);
    
    return final_sum;
}