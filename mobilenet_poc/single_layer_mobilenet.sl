// Single Layer MobileNet with Real Weights
include "simptensor/tensor_core.sl";

fn single_layer_conv(f32[] weights, i32 weight_count) -> f32 {
    // Input: simplified to single pixel for testing
    var input_val = 0.5;  // Normalized RGB pixel value
    
    // First layer: Conv 3x3, stride=2, pad=1 (3 -> 32 channels)
    // Use actual weights from ONNX model
    var conv_result = 0.0;
    
    // Simple convolution: input * first weight + bias approximation
    if (weight_count > 0) {
        conv_result = input_val * weights[0];  // Use real conv weight
    }
    
    // BatchNorm: use weights at offset 864
    var bn_offset = 864;
    if (weight_count > bn_offset) {
        var gamma = weights[bn_offset];          // gamma (scale)
        var beta = weights[bn_offset + 32];      // beta (shift) 
        var mean = weights[bn_offset + 64];      // running mean
        var variance = weights[bn_offset + 96];  // running var
        
        // BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
        var eps = 0.00001;
        var norm_result = (conv_result - mean) / (variance + eps) * gamma + beta;
        conv_result = norm_result;
    }
    
    // ReLU activation
    if (conv_result < 0.0) {
        conv_result = 0.0;
    }
    
    return conv_result;
}

fn kernel_main() -> f32 {
    // Fallback without weights
    return 0.0;
}

fn kernel_main_with_weights(f32[] weights, i32 weight_count) -> f32 {
    return single_layer_conv(weights, weight_count);
}
