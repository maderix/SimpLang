// SimpBLAS Integration Example
// Demonstrates calling optimized SimpBLAS functions from SimpleLang

include "../simptensor/tensor_core.sl";

// Simple matrix operations using SimpBLAS-style computation
fn test_matrix_multiply() -> f32 {
    // 2x2 matrix multiplication: C = A * B
    // A = [1, 2]    B = [5, 6]    Expected C = [19, 22]
    //     [3, 4]        [7, 8]                   [43, 50]
    
    var M = 2;
    var N = 2; 
    var K = 2;
    
    // Create matrices using SIMD arrays
    var A = array<f32, simd=auto>([4]);
    var B = array<f32, simd=auto>([4]);
    var C = array<f32, simd=auto>([4]);
    
    // Initialize matrix A
    A[0] = 1.0; A[1] = 2.0;
    A[2] = 3.0; A[3] = 4.0;
    
    // Initialize matrix B
    B[0] = 5.0; B[1] = 6.0;
    B[2] = 7.0; B[3] = 8.0;
    
    // Manual matrix multiplication (demonstrates computation pattern)
    // C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 19
    C[0] = A[0] * B[0] + A[1] * B[2];
    
    // C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 22
    C[1] = A[0] * B[1] + A[1] * B[3];
    
    // C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 3*5 + 4*7 = 43
    C[2] = A[2] * B[0] + A[3] * B[2];
    
    // C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 3*6 + 4*8 = 50
    C[3] = A[2] * B[1] + A[3] * B[3];
    
    // Return sum for verification: 19 + 22 + 43 + 50 = 134
    return C[0] + C[1] + C[2] + C[3];
}

fn test_element_wise_operations() -> f32 {
    var size = 8;
    
    // Create vectors using SIMD arrays
    var A = array<f32, simd=avx>([size]);
    var B = array<f32, simd=avx>([size]);
    var C = array<f32, simd=avx>([size]);
    var result = array<f32, simd=avx>([size]);
    
    // Initialize vectors
    var i = 0;
    while (i < size) {
        A[i] = i * 0.5;          // [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        B[i] = (i + 1) * 0.25;   // [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        C[i] = 2.0;              // [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        i = i + 1;
    }
    
    // Element-wise operations: result = (A + B) * C
    i = 0;
    var sum = 0.0;
    while (i < size) {
        var temp = A[i] + B[i];  // Element-wise addition
        result[i] = temp * C[i]; // Element-wise multiplication
        sum = sum + result[i];
        i = i + 1;
    }
    
    // Apply ReLU6-like activation
    i = 0;
    while (i < size) {
        if (result[i] < 0.0) {
            result[i] = 0.0;
        }
        if (result[i] > 6.0) {
            result[i] = 6.0;
        }
        i = i + 1;
    }
    
    // Return sum for verification
    var final_sum = 0.0;
    i = 0;
    while (i < size) {
        final_sum = final_sum + result[i];
        i = i + 1;
    }
    
    return final_sum;
}

fn test_convolution_pattern() -> f32 {
    // Simplified 2D convolution pattern
    // Input: 3x3, Kernel: 2x2, Output: 2x2
    
    var input_size = 9;   // 3x3 input
    var kernel_size = 4;  // 2x2 kernel
    var output_size = 4;  // 2x2 output
    
    var input = array<f32>([input_size]);
    var kernel = array<f32>([kernel_size]);
    var output = array<f32>([output_size]);
    
    // Initialize 3x3 input
    input[0] = 1.0; input[1] = 2.0; input[2] = 3.0;
    input[3] = 4.0; input[4] = 5.0; input[5] = 6.0;
    input[6] = 7.0; input[7] = 8.0; input[8] = 9.0;
    
    // Initialize 2x2 kernel  
    kernel[0] = 0.1; kernel[1] = 0.2;
    kernel[2] = 0.3; kernel[3] = 0.4;
    
    // Convolution: slide kernel over input
    // Output[0,0] = input[0:2,0:2] * kernel
    output[0] = input[0] * kernel[0] + input[1] * kernel[1] + 
                input[3] * kernel[2] + input[4] * kernel[3];
    
    // Output[0,1] = input[0:2,1:3] * kernel  
    output[1] = input[1] * kernel[0] + input[2] * kernel[1] + 
                input[4] * kernel[2] + input[5] * kernel[3];
    
    // Output[1,0] = input[1:3,0:2] * kernel
    output[2] = input[3] * kernel[0] + input[4] * kernel[1] + 
                input[6] * kernel[2] + input[7] * kernel[3];
    
    // Output[1,1] = input[1:3,1:3] * kernel
    output[3] = input[4] * kernel[0] + input[5] * kernel[1] + 
                input[7] * kernel[2] + input[8] * kernel[3];
    
    // Return sum for verification
    return output[0] + output[1] + output[2] + output[3];
}

fn kernel_main() -> f32 {
    // Test all SimpBLAS-style operations
    var gemm_result = test_matrix_multiply();        // Expected: 134.0
    var elem_result = test_element_wise_operations(); // Variable based on calculations
    var conv_result = test_convolution_pattern();     // Variable based on calculations
    
    // Combine results for comprehensive test
    return gemm_result + elem_result * 0.01 + conv_result * 0.001;
}