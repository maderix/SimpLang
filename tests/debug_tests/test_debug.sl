fn add_vectors(var a SSESlice, var b SSESlice) {
    // Load vectors
    var vec1 = slice_get_sse(a, 0i);
    var vec2 = slice_get_sse(b, 0i);
    
    // Add vectors
    var result = simd_add(vec1, vec2);
    return result;
}

fn multiply_vectors(var a SSESlice, var b SSESlice) {
    // Load vectors
    var vec1 = slice_get_sse(a, 0i);
    var vec2 = slice_get_sse(b, 0i);
    
    // Multiply vectors
    var result = simd_mul(vec1, vec2);
    return result;
}

fn kernel_main() {
    // Create test vectors
    var input1 = make(SSESlice, 4);
    var input2 = make(SSESlice, 4);
    var output = make(SSESlice, 4);

    // Initialize vectors
    input1[0i] = sse(1.0, 2.0, 3.0, 4.0);
    input2[0i] = sse(5.0, 6.0, 7.0, 8.0);

    // Test vector addition
    var sum = add_vectors(input1, input2);
    slice_set_sse(output, 0i, sum);

    // Test vector multiplication
    var product = multiply_vectors(input1, input2);
    slice_set_sse(output, 1i, product);

    return 1.0;  // Success
}