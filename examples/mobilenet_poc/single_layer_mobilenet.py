#!/usr/bin/env python3

import onnx
import numpy as np

def generate_single_layer_mobilenet():
    """Generate single MobileNet layer with proper weight loading"""
    
    # Load the ONNX model
    model = onnx.load('mobilenetv2-7.onnx')
    
    # Get first Conv layer (mobilenetv20_features_conv0_fwd)
    first_conv = model.graph.node[0]  # Conv 3x3 stride=2, 3->32 channels
    first_bn = model.graph.node[1]    # BatchNormalization 
    first_relu = model.graph.node[2]  # ReLU
    
    # Get weight data
    weight_map = {}
    for initializer in model.graph.initializer:
        if initializer.raw_data:
            weight_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
        else:
            weight_data = np.array(initializer.float_data, dtype=np.float32)
        weight_map[initializer.name] = weight_data
    
    # Extract actual weights for first layer
    conv_weights = weight_map['mobilenetv20_features_conv0_weight']  # [32, 3, 3, 3]
    bn_gamma = weight_map['mobilenetv20_features_batchnorm0_gamma']   # [32]
    bn_beta = weight_map['mobilenetv20_features_batchnorm0_beta']     # [32]
    bn_mean = weight_map['mobilenetv20_features_batchnorm0_running_mean']  # [32]
    bn_var = weight_map['mobilenetv20_features_batchnorm0_running_var']    # [32]
    
    print(f"Conv weights shape: {conv_weights.shape}")
    print(f"BN gamma shape: {bn_gamma.shape}")
    print(f"First few conv weights: {conv_weights[:5]}")
    print(f"First few BN gamma: {bn_gamma[:5]}")
    
    code = f"""// Single Layer MobileNet with Real Weights
include "simptensor/tensor_core.sl";

fn single_layer_conv(f32[] weights, i32 weight_count) -> f32 {{
    // Input: simplified to single pixel for testing
    var input_val = 0.5;  // Normalized RGB pixel value
    
    // First layer: Conv 3x3, stride=2, pad=1 (3 -> 32 channels)
    // Use actual weights from ONNX model
    var conv_result = 0.0;
    
    // Simple convolution: input * first weight + bias approximation
    if (weight_count > 0) {{
        conv_result = input_val * weights[0];  // Use real conv weight
    }}
    
    // BatchNorm: use weights at offset {len(conv_weights)}
    var bn_offset = {len(conv_weights)};
    if (weight_count > bn_offset) {{
        var gamma = weights[bn_offset];          // gamma (scale)
        var beta = weights[bn_offset + 32];      // beta (shift) 
        var mean = weights[bn_offset + 64];      // running mean
        var variance = weights[bn_offset + 96];  // running var
        
        // BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
        var eps = 0.00001;
        var norm_result = (conv_result - mean) / (variance + eps) * gamma + beta;
        conv_result = norm_result;
    }}
    
    // ReLU activation
    if (conv_result < 0.0) {{
        conv_result = 0.0;
    }}
    
    return conv_result;
}}

fn kernel_main() -> f32 {{
    // Fallback without weights
    return 0.0;
}}

fn kernel_main_with_weights(f32[] weights, i32 weight_count) -> f32 {{
    return single_layer_conv(weights, weight_count);
}}
"""
    
    return code, len(conv_weights) + len(bn_gamma) + len(bn_beta) + len(bn_mean) + len(bn_var)

def main():
    code, total_weights = generate_single_layer_mobilenet()
    
    with open('single_layer_mobilenet.sl', 'w') as f:
        f.write(code)
    
    print(f"Generated single_layer_mobilenet.sl")
    print(f"Total weights needed: {total_weights}")
    print("This kernel expects weights in order: [conv_weights, bn_gamma, bn_beta, bn_mean, bn_var]")

if __name__ == '__main__':
    main()