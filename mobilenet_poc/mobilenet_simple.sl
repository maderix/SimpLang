// Simple 3-Layer MobileNet Test with SimpBLAS Integration  
include "simptensor/tensor_core.sl";

// SimpBLAS functions will be linked externally at compile time

fn mobilenet_3_layer_test(f32[] weights, i32 weight_count) -> f32 {
    // Initialize SimpBLAS first
    var blas_init = sb_init();
    
    // Input: 1x224x224x3
    var input_size = 150528;
    var input_data = array<f32, simd=auto>([input_size]);
    
    var i = 0;
    while (i < input_size) {
        input_data[i] = 0.5;
        i = i + 1;
    }
    
    // Layer 1: Simple conv using global weights
    var conv1_size = 401408;  // 112*112*32
    var conv1_out = array<f32, simd=auto>([conv1_size]);
    
    var conv_i = 0;
    while (conv_i < conv1_size) {
        var weight_idx = conv_i % weight_count;
        if (weight_idx < weight_count) {
            conv1_out[conv_i] = input_data[conv_i % input_size] * weights[weight_idx];
        } else {
            conv1_out[conv_i] = input_data[conv_i % input_size] * 0.01;
        }
        conv_i = conv_i + 1;
    }
    
    // Layer 2: BatchNorm + ReLU
    var bn_i = 0;
    while (bn_i < conv1_size) {
        var val = conv1_out[bn_i];
        
        // Simple batch norm with passed weights
        var param_idx = (bn_i % 32) + 1000;  // Use weights at offset 1000+
        if (param_idx + 32 < weight_count) {
            val = val * weights[param_idx] + weights[param_idx + 32];
        }
        
        // ReLU
        if (val < 0.0) {
            val = 0.0;
        }
        
        conv1_out[bn_i] = val;
        bn_i = bn_i + 1;
    }
    
    // Layer 3: Simple GEMM test (2x2 matrix multiply)
    // A = [1, 2]  B = [5, 6]  Expected C = [19, 22]
    //     [3, 4]      [7, 8]                 [43, 50]
    var A = array<f32, simd=auto>([4]);
    var B = array<f32, simd=auto>([4]);
    var C = array<f32, simd=auto>([4]);
    
    // Initialize test matrices
    A[0] = 1.0; A[1] = 2.0; A[2] = 3.0; A[3] = 4.0;
    B[0] = 5.0; B[1] = 6.0; B[2] = 7.0; B[3] = 8.0;
    C[0] = 0.0; C[1] = 0.0; C[2] = 0.0; C[3] = 0.0;
    
    // Call SimpBLAS GEMM: C = A * B (2x2 matrices)
    sb_gemm_f32(2, 2, 2, A, 2, B, 2, C, 2);
    
    // Expected result: C = [19, 22, 43, 50], sum = 134
    var gemm_result = C[0] + C[1] + C[2] + C[3];
    
    // Return combined result
    var sum = 0.0;
    var avg_i = 0;
    while (avg_i < 100) {
        sum = sum + conv1_out[avg_i];
        avg_i = avg_i + 1;
    }
    
    return (sum / 100.0) + gemm_result * 0.01;
}

fn kernel_main(f32[] weights, i32 weight_count) -> f32 {
    return mobilenet_3_layer_test(weights, weight_count);
}
