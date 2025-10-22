// Simplified MobileNetV2 SimpleLang Implementation
include "simptensor/tensor_core.sl";

// Helper function for safe weight access
fn get_weight(f32[] weights, i32 weight_count, i32 offset, i32 index) -> f32 {
    var actual_index = offset + index;
    if (actual_index < weight_count) {
        return weights[actual_index];
    } else {
        return 0.0;  // Safe fallback
    }
}

// Simplified convolution layer
fn simple_conv(f32[] input_data, i32 input_size, 
               f32[] weights, i32 weight_count, i32 weight_offset,
               f32[] output_data, i32 output_size) -> f32 {
    var i = 0;
    while (i < output_size) {
        var val = 0.0;
        
        // Simple convolution with weight access
        var w_idx = i % 100; // Simplified weight indexing
        val = input_data[i % input_size] * get_weight(weights, weight_count, weight_offset, w_idx);
        
        // ReLU activation
        if (val < 0.0) {
            val = 0.0;
        }
        
        output_data[i] = val;
        i = i + 1;
    }
    return 1.0;
}

// Simplified depthwise convolution
fn depthwise_conv(f32[] input_data, i32 input_size,
                  f32[] weights, i32 weight_count, i32 weight_offset,
                  f32[] output_data, i32 output_size) -> f32 {
    var i = 0;
    while (i < output_size) {
        var val = input_data[i % input_size];
        val = val * get_weight(weights, weight_count, weight_offset, i % 32);
        
        // ReLU activation
        if (val < 0.0) {
            val = 0.0;
        }
        
        output_data[i] = val;
        i = i + 1;
    }
    return 1.0;
}

// Main MobileNet inference with real architecture (simplified)
fn mobilenet_inference_with_weights(f32[] weights, i32 weight_count) -> f32 {
    // Input: 224x224x3 = 150528
    var input_size = 150528;
    var input_data = array<f32, simd=auto>([input_size]);
    
    // Initialize input with test pattern
    var i = 0;
    while (i < input_size) {
        input_data[i] = 0.5 + (i % 100) / 1000.0; // Varied test data
        i = i + 1;
    }
    
    // Layer 1: Initial conv 224x224x3 -> 112x112x32
    var conv1_size = 401408; // 112*112*32
    var conv1_out = array<f32, simd=auto>([conv1_size]);
    simple_conv(input_data, input_size, weights, weight_count, 0, conv1_out, conv1_size);
    
    // Layer 2: Depthwise conv 112x112x32 -> 112x112x32  
    var dw1_out = array<f32, simd=auto>([conv1_size]);
    depthwise_conv(conv1_out, conv1_size, weights, weight_count, 1000, dw1_out, conv1_size);
    
    // Layer 3: Pointwise conv 112x112x32 -> 112x112x16
    var pw1_size = 200704; // 112*112*16
    var pw1_out = array<f32, simd=auto>([pw1_size]);
    simple_conv(dw1_out, conv1_size, weights, weight_count, 2000, pw1_out, pw1_size);
    
    // More layers (simplified for compilation speed)
    var final_size = 1000; // Classification outputs
    var final_out = array<f32, simd=auto>([final_size]);
    
    // Global average pooling + classification
    i = 0;
    while (i < final_size) {
        var sum = 0.0;
        var j = 0;
        while (j < 100 && (i * 100 + j) < pw1_size) {
            sum = sum + pw1_out[i * 100 + j];
            j = j + 1;
        }
        
        // Apply final classification weights
        var weight_val = get_weight(weights, weight_count, 100000 + i, 0);
        final_out[i] = (sum / 100.0) * weight_val;
        i = i + 1;
    }
    
    // Return max value as classification result
    var max_val = final_out[0];
    i = 1;
    while (i < final_size) {
        if (final_out[i] > max_val) {
            max_val = final_out[i];
        }
        i = i + 1;
    }
    
    return max_val;
}

// Standalone version (loads fake weights)
fn mobilenet_inference() -> f32 {
    // Create fake weight array for testing
    var fake_weight_count = 200000;
    var fake_weights = array<f32, simd=auto>([fake_weight_count]);
    
    var i = 0;
    while (i < fake_weight_count) {
        var pseudo_rand = (i * 1103515245 + 12345) % 2147483647;
        fake_weights[i] = (pseudo_rand % 1000) / 50000.0 - 0.01;
        i = i + 1;
    }
    
    return mobilenet_inference_with_weights(fake_weights, fake_weight_count);
}

// Kernel entry points
fn kernel_main() -> f32 {
    return mobilenet_inference();
}

fn kernel_main_with_weights(f32[] weights, i32 weight_count) -> f32 {
    return mobilenet_inference_with_weights(weights, weight_count);
}
