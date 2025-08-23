fn kernel_main() {
    var size = 10000;
    var iterations = 1000;
    
    // Test 1: Large array creation and initialization
    var large_array = array<f32>([size]);
    var i = 0;
    while (i < size) {
        large_array[i] = i * 2.5;
        i = i + 1;
    }
    
    // Test 2: Sequential access pattern (cache-friendly)
    var sum = 0.0;
    var iter = 0;
    while (iter < iterations) {
        i = 0;
        while (i < size) {
            sum = sum + large_array[i];
            i = i + 1;
        }
        iter = iter + 1;
    }
    
    // Test 3: Random access pattern (cache-unfriendly) - simplified
    var random_sum = 0.0;
    iter = 0;
    while (iter < iterations) {
        i = 0;
        while (i < size) {
            var random_idx = (size - 1) - i;  // Simple reverse pattern
            random_sum = random_sum + large_array[random_idx];
            i = i + 1;
        }
        iter = iter + 1;
    }
    
    // Test 4: Array-to-array operations
    var array2 = array<f32>([size]);
    var array3 = array<f32>([size]);
    
    // Initialize arrays
    i = 0;
    while (i < size) {
        array2[i] = i * 1.5;
        array3[i] = i * 0.5;
        i = i + 1;
    }
    
    // Element-wise addition
    var result_sum = 0.0;
    iter = 0;
    while (iter < iterations) {
        i = 0;
        while (i < size) {
            var temp_result = array2[i] + array3[i] + large_array[i];
            result_sum = result_sum + temp_result;
            i = i + 1;
        }
        iter = iter + 1;
    }
    
    // Test 5: Stride access patterns - simplified
    var stride_sum = 0.0;
    iter = 0;
    while (iter < iterations) {
        i = 0;
        while (i < size) {
            stride_sum = stride_sum + large_array[i];  // Just sequential access
            i = i + 1;
        }
        iter = iter + 1;
    }
    
    // Test 6: Reverse access pattern
    var reverse_sum = 0.0;
    iter = 0;
    while (iter < iterations) {
        i = size - 1;
        while (i >= 0) {
            reverse_sum = reverse_sum + large_array[i];
            i = i - 1;
        }
        iter = iter + 1;
    }
    
    // Return combined result to prevent optimization
    return sum + random_sum + result_sum + stride_sum + reverse_sum;
}