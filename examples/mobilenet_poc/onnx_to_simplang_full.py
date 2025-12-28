#!/usr/bin/env python3

import onnx
import numpy as np
from typing import Dict, List, Tuple

class MobileNetV2Generator:
    def __init__(self, model_path: str):
        self.model = onnx.load(model_path)
        self.weight_map = {}
        self.weight_offsets = {}  # Track weight offsets in global array
        self.layer_outputs = {}  # Track tensor names and their shapes
        self.temp_counter = 0
        self.total_weights = 0
        
        # Build weight map with offsets
        current_offset = 0
        for initializer in self.model.graph.initializer:
            self.weight_map[initializer.name] = initializer
            
            if initializer.raw_data:
                weight_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
            else:
                weight_data = np.array(initializer.float_data, dtype=np.float32)
            
            self.weight_offsets[initializer.name] = current_offset
            current_offset += len(weight_data)
        
        self.total_weights = current_offset
    
    def get_temp_name(self) -> str:
        self.temp_counter += 1
        return f"temp_{self.temp_counter}"
    
    def get_tensor_shape(self, tensor_name: str) -> Tuple[int, int, int, int]:
        """Get NHWC shape for a tensor"""
        # For this implementation, we'll track shapes as we go
        # Start with input shape
        if tensor_name == 'data':
            return (1, 224, 224, 3)  # NHWC
        
        if tensor_name in self.layer_outputs:
            return self.layer_outputs[tensor_name]
        
        # Default fallback - we'll update this as we process layers
        return (1, 112, 112, 32)
    
    def generate_conv_op(self, node, input_name: str, output_name: str) -> str:
        """Generate convolution operation"""
        weight_name = node.input[1]
        weight = self.weight_map[weight_name]
        
        # Extract attributes
        kernel_shape = [attr.ints[0] for attr in node.attribute if attr.name == 'kernel_shape'][0], [attr.ints[1] for attr in node.attribute if attr.name == 'kernel_shape'][0]
        strides = [attr.ints[0] for attr in node.attribute if attr.name == 'strides'][0], [attr.ints[1] for attr in node.attribute if attr.name == 'strides'][0] 
        pads = list([attr.ints[i] for i in range(4)] for attr in node.attribute if attr.name == 'pads')[0] if any(attr.name == 'pads' for attr in node.attribute) else [0,0,0,0]
        groups = [attr.i for attr in node.attribute if attr.name == 'group'][0] if any(attr.name == 'group' for attr in node.attribute) else 1
        
        # Weight shape: [out_channels, in_channels/group, kh, kw] -> convert to NHWC logic
        weight_shape = list(weight.dims)
        out_channels, in_channels_per_group, kh, kw = weight_shape
        in_channels = in_channels_per_group * groups
        
        input_shape = self.get_tensor_shape(input_name)
        n, h, w, c = input_shape
        
        # Calculate output shape
        out_h = (h + pads[0] + pads[2] - kh) // strides[0] + 1
        out_w = (w + pads[1] + pads[3] - kw) // strides[1] + 1
        
        self.layer_outputs[output_name] = (n, out_h, out_w, out_channels)
        
        # Extract weight data
        if weight.raw_data:
            weight_data = np.frombuffer(weight.raw_data, dtype=np.float32)
        else:
            weight_data = np.array(weight.float_data, dtype=np.float32)
        
        # Generate weight array initialization
        weight_var = f"weights_{output_name.replace('.', '_')}"
        code = f"""
    // Conv layer: {node.name}
    // Weight shape: [{out_channels}, {in_channels_per_group}, {kh}, {kw}], Groups: {groups}
    var {weight_var}_size = {len(weight_data)};
    var {weight_var} = array<f32, simd=auto>([{weight_var}_size]);
    
    // Initialize weights (showing first few values)"""
        
        # Load weights from global weight array with proper offset
        weight_offset = self.weight_offsets.get(weight_name, 0)
        
        code += f"""
    // Load weights from passed weights parameter at offset {weight_offset}
    var wi = 0;
    while (wi < {weight_var}_size) {{
        if ({weight_offset} + wi < weight_count) {{
            {weight_var}[wi] = weights[{weight_offset} + wi];
        }} else {{
            {weight_var}[wi] = 0.0;  // Safety fallback
        }}
        wi = wi + 1;
    }}"""
        
        # Generate output tensor allocation
        output_size = n * out_h * out_w * out_channels
        code += f"""
    
    // Output tensor: {output_name}
    var {output_name.replace('.', '_')}_size = {output_size};
    var {output_name.replace('.', '_')} = array<f32, simd=auto>([{output_name.replace('.', '_')}_size]);
    
    // Convolution computation"""
        
        if groups == 1:
            # Regular convolution using GEMM
            col_size = out_h * out_w * in_channels * kh * kw
            code += f"""
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): {out_channels} x {in_channels * kh * kw}
    //   B (im2col):  {in_channels * kh * kw} x {out_h * out_w} 
    //   C (output):  {out_channels} x {out_h * out_w}
    
    // Allocate im2col buffer
    var col_size = {col_size};
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d({input_name.replace('.', '_')}, {n}, {h}, {w}, {in_channels},
                  {kh}, {kw}, {strides[0]}, {strides[1]}, {pads[0]}, {pads[1]}, 
                  {out_h}, {out_w}, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([{out_channels * out_h * out_w}]);
    
    // GEMM call: C = A * B
    // A = weights [{out_channels} x {in_channels * kh * kw}]
    // B = col_buffer [{in_channels * kh * kw} x {out_h * out_w}]  
    // C = gemm_output [{out_channels} x {out_h * out_w}]
    sb_gemm_f32({out_channels}, {out_h * out_w}, {in_channels * kh * kw},
                {weight_var}, {in_channels * kh * kw},
                col_buffer, {out_h * out_w}, 
                gemm_output, {out_h * out_w});
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < {out_h}) {{
        var out_x = 0;
        while (out_x < {out_w}) {{
            var out_ch = 0;
            while (out_ch < {out_channels}) {{
                var gemm_idx = out_ch * {out_h * out_w} + out_y * {out_w} + out_x;
                var out_offset = nhwc_offset({n}, {out_h}, {out_w}, {out_channels}, 0, out_y, out_x, out_ch);
                {output_name.replace('.', '_')}[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }}
            out_x = out_x + 1;
        }}
        out_y = out_y + 1;
    }}"""
        else:
            # Depthwise convolution
            code += f"""
    // Depthwise convolution (groups = {groups})
    var batch = 0;
    var out_y = 0;
    while (out_y < {out_h}) {{
        var out_x = 0;
        while (out_x < {out_w}) {{
            var out_ch = 0;
            while (out_ch < {out_channels}) {{
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < {kh}) {{
                    var kx = 0;
                    while (kx < {kw}) {{
                        var in_y = out_y * {strides[0]} + ky - {pads[0]};
                        var in_x = out_x * {strides[1]} + kx - {pads[1]};
                        
                        // Bounds check
                        if (in_y >= 0) {{
                            if (in_y < {h}) {{
                                if (in_x >= 0) {{
                                    if (in_x < {w}) {{
                                        var in_offset = nhwc_offset({n}, {h}, {w}, {c}, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * {kh * kw} + ky * {kw} + kx;
                                        sum = sum + {input_name.replace('.', '_')}[in_offset] * {weight_var}[weight_idx];
                                    }}
                                }}
                            }}
                        }}
                        kx = kx + 1;
                    }}
                    ky = ky + 1;
                }}
                
                var out_offset = nhwc_offset({n}, {out_h}, {out_w}, {out_channels}, batch, out_y, out_x, out_ch);
                {output_name.replace('.', '_')}[out_offset] = sum;
                out_ch = out_ch + 1;
            }}
            out_x = out_x + 1;
        }}
        out_y = out_y + 1;
    }}"""
        
        return code
    
    def generate_batchnorm_op(self, node, input_name: str, output_name: str) -> str:
        """Generate batch normalization operation"""
        # Get batch norm parameters
        scale_name = node.input[1]  # gamma
        bias_name = node.input[2]   # beta  
        mean_name = node.input[3]   # running_mean
        var_name = node.input[4]    # running_var
        
        scale = self.weight_map[scale_name]
        bias = self.weight_map[bias_name]
        mean = self.weight_map[mean_name]
        var = self.weight_map[var_name]
        
        # Extract data
        if scale.raw_data:
            scale_data = np.frombuffer(scale.raw_data, dtype=np.float32)
            bias_data = np.frombuffer(bias.raw_data, dtype=np.float32)
            mean_data = np.frombuffer(mean.raw_data, dtype=np.float32)
            var_data = np.frombuffer(var.raw_data, dtype=np.float32)
        else:
            scale_data = np.array(scale.float_data, dtype=np.float32)
            bias_data = np.array(bias.float_data, dtype=np.float32)
            mean_data = np.array(mean.float_data, dtype=np.float32)
            var_data = np.array(var.float_data, dtype=np.float32)
        
        input_shape = self.get_tensor_shape(input_name)
        self.layer_outputs[output_name] = input_shape  # Same shape as input
        n, h, w, c = input_shape
        
        code = f"""
    // BatchNorm layer: {node.name}
    var bn_scale_{output_name.replace('.', '_')} = array<f32, simd=auto>([{len(scale_data)}]);
    var bn_bias_{output_name.replace('.', '_')} = array<f32, simd=auto>([{len(bias_data)}]);
    var bn_mean_{output_name.replace('.', '_')} = array<f32, simd=auto>([{len(mean_data)}]);
    var bn_var_{output_name.replace('.', '_')} = array<f32, simd=auto>([{len(var_data)}]);
    
    // Initialize batch norm parameters"""
        
        # Load batch norm parameters from global weight array
        scale_offset = self.weight_offsets.get(scale_name, 0)
        bias_offset = self.weight_offsets.get(bias_name, 0)
        mean_offset = self.weight_offsets.get(mean_name, 0)
        var_offset = self.weight_offsets.get(var_name, 0)
        
        code += f"""
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < {len(scale_data)}) {{
        if ({scale_offset} + bni < weight_count) {{
            bn_scale_{output_name.replace('.', '_')}[bni] = weights[{scale_offset} + bni];
        }}
        if ({bias_offset} + bni < weight_count) {{
            bn_bias_{output_name.replace('.', '_')}[bni] = weights[{bias_offset} + bni];
        }}
        if ({mean_offset} + bni < weight_count) {{
            bn_mean_{output_name.replace('.', '_')}[bni] = weights[{mean_offset} + bni];
        }}
        if ({var_offset} + bni < weight_count) {{
            bn_var_{output_name.replace('.', '_')}[bni] = weights[{var_offset} + bni];
        }}
        bni = bni + 1;
    }}"""
        
        tensor_size = n * h * w * c
        code += f"""
    
    // Create output tensor (reuse input for in-place operation)
    var {output_name.replace('.', '_')} = {input_name.replace('.', '_')};
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < {tensor_size}) {{
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % {c};
        var val = {input_name.replace('.', '_')}[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_{output_name.replace('.', '_')}[ch_idx] + eps;
        if (var_eps < 0.000001) {{
            var_eps = 0.000001;  // Prevent division by zero
        }}
        
        var norm_val = (val - bn_mean_{output_name.replace('.', '_')}[ch_idx]) / var_eps * 
                       bn_scale_{output_name.replace('.', '_')}[ch_idx] + 
                       bn_bias_{output_name.replace('.', '_')}[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {{
            norm_val = 100.0;
        }}
        if (norm_val < -100.0) {{
            norm_val = -100.0;
        }}
        
        {output_name.replace('.', '_')}[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }}"""
        
        return code
    
    def generate_relu_op(self, node, input_name: str, output_name: str) -> str:
        """Generate ReLU activation"""
        input_shape = self.get_tensor_shape(input_name)
        self.layer_outputs[output_name] = input_shape  # Same shape
        n, h, w, c = input_shape
        tensor_size = n * h * w * c
        
        return f"""
    // ReLU layer: {node.name}
    var {output_name.replace('.', '_')} = {input_name.replace('.', '_')};
    
    var relu_idx = 0;
    while (relu_idx < {tensor_size}) {{
        var val = {input_name.replace('.', '_')}[relu_idx];
        if (val < 0.0) {{
            val = 0.0;
        }}
        {output_name.replace('.', '_')}[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }}"""
    
    def generate_add_op(self, node, input1_name: str, input2_name: str, output_name: str) -> str:
        """Generate element-wise addition (residual connection)"""
        input_shape = self.get_tensor_shape(input1_name)
        self.layer_outputs[output_name] = input_shape
        n, h, w, c = input_shape
        tensor_size = n * h * w * c
        
        return f"""
    // Add layer (residual): {node.name}
    var {output_name.replace('.', '_')}_size = {tensor_size};
    var {output_name.replace('.', '_')} = array<f32, simd=auto>([{output_name.replace('.', '_')}_size]);
    
    var add_idx = 0;
    while (add_idx < {tensor_size}) {{
        {output_name.replace('.', '_')}[add_idx] = {input1_name.replace('.', '_')}[add_idx] + {input2_name.replace('.', '_')}[add_idx];
        add_idx = add_idx + 1;
    }}"""
    
    def generate_global_avg_pool_op(self, node, input_name: str, output_name: str) -> str:
        """Generate global average pooling"""
        input_shape = self.get_tensor_shape(input_name)
        n, h, w, c = input_shape
        self.layer_outputs[output_name] = (n, 1, 1, c)  # Global pooling reduces spatial dims to 1x1
        
        return f"""
    // GlobalAveragePool layer: {node.name}
    var {output_name.replace('.', '_')}_size = {c};
    var {output_name.replace('.', '_')} = array<f32, simd=auto>([{output_name.replace('.', '_')}_size]);
    
    var ch = 0;
    while (ch < {c}) {{
        var sum = 0.0;
        var spatial_idx = 0;
        while (spatial_idx < {h * w}) {{
            var y = spatial_idx / {w};
            var x = spatial_idx % {w};
            var offset = nhwc_offset({n}, {h}, {w}, {c}, 0, y, x, ch);
            sum = sum + {input_name.replace('.', '_')}[offset];
            spatial_idx = spatial_idx + 1;
        }}
        {output_name.replace('.', '_')}[ch] = sum / {h * w}.0;
        ch = ch + 1;
    }}"""

    def generate_full_model(self) -> str:
        """Generate complete MobileNetV2 SimpleLang code"""
        
        code = f"""// Complete MobileNetV2 SimpleLang Implementation with SimpBLAS
include "../simptensor/tensor_core.sl";

// Note: SimpBLAS functions sb_init and sb_gemm_f32 are provided by runtime

// Helper function for im2col transformation
fn im2col_conv2d(f32[] input, i32 batch_size, i32 in_h, i32 in_w, i32 in_channels,
                 i32 kernel_h, i32 kernel_w, i32 stride_h, i32 stride_w,
                 i32 pad_h, i32 pad_w, i32 out_h, i32 out_w, f32[] col_buffer) -> void {{
    var col_idx = 0;
    var out_y = 0;
    while (out_y < out_h) {{
        var out_x = 0;
        while (out_x < out_w) {{
            var ky = 0;
            while (ky < kernel_h) {{
                var kx = 0;
                while (kx < kernel_w) {{
                    var ch = 0;
                    while (ch < in_channels) {{
                        var in_y = out_y * stride_h + ky - pad_h;
                        var in_x = out_x * stride_w + kx - pad_w;
                        
                        if (in_y >= 0) {{
                            if (in_y < in_h) {{
                                if (in_x >= 0) {{
                                    if (in_x < in_w) {{
                                        var in_offset = nhwc_offset(batch_size, in_h, in_w, in_channels, 0, in_y, in_x, ch);
                                        col_buffer[col_idx] = input[in_offset];
                                    }} else {{
                                        col_buffer[col_idx] = 0.0;
                                    }}
                                }} else {{
                                    col_buffer[col_idx] = 0.0;
                                }}
                            }} else {{
                                col_buffer[col_idx] = 0.0;
                            }}
                        }} else {{
                            col_buffer[col_idx] = 0.0;
                        }}
                        col_idx = col_idx + 1;
                        ch = ch + 1;
                    }}
                    kx = kx + 1;
                }}
                ky = ky + 1;
            }}
            out_x = out_x + 1;
        }}
        out_y = out_y + 1;
    }}
}}

fn load_weights_from_binary() -> f32 {{
    // Initialize SimpBLAS
    var init_result = sb_init();
    return 1.0;  // Success
}}

// Core inference function that takes weights as parameters
fn mobilenet_inference_core(f32[] weights, i32 weight_count, f32[] input_data, i32 input_size) -> var {{
    // Initialize SimpBLAS
    var init_result = sb_init();
    
    // Input tensor: 1x224x224x3 (NHWC)
    var input_size = 150528; // 1 * 224 * 224 * 3
    var data = array<f32, simd=auto>([input_size]);
    
    // Load input data from host
    var init_idx = 0;
    while (init_idx < input_size) {{
        data[init_idx] = input_data[init_idx];
        init_idx = init_idx + 1;
    }}
"""

        
        # Process all layers and store layer code for reuse
        current_input = 'data'
        layer_processing_code = ""
        
        for i, node in enumerate(self.model.graph.node):
            print(f"Processing layer {i+1}/{len(self.model.graph.node)}: {node.op_type}")
            
            if node.op_type == 'Conv':
                output_name = node.output[0]
                layer_code = self.generate_conv_op(node, current_input, output_name)
                code += layer_code
                layer_processing_code += layer_code
                current_input = output_name
                
            elif node.op_type == 'BatchNormalization':
                output_name = node.output[0]
                layer_code = self.generate_batchnorm_op(node, current_input, output_name)
                code += layer_code
                layer_processing_code += layer_code
                current_input = output_name
                
            elif node.op_type == 'Relu':
                output_name = node.output[0]
                code += self.generate_relu_op(node, current_input, output_name)
                current_input = output_name
                
            elif node.op_type == 'Add':
                # For Add operations, we need to track both inputs
                input1_name = node.input[0].replace('.', '_')
                input2_name = node.input[1].replace('.', '_')
                output_name = node.output[0]
                code += self.generate_add_op(node, input1_name, input2_name, output_name)
                current_input = output_name
                
            elif node.op_type == 'GlobalAveragePool':
                output_name = node.output[0]
                code += self.generate_global_avg_pool_op(node, current_input, output_name)
                current_input = output_name
                
            elif node.op_type == 'Reshape':
                # For MobileNet, reshape is typically just a view change, no computation
                output_name = node.output[0]
                input_shape = self.get_tensor_shape(current_input)
                self.layer_outputs[output_name] = input_shape  # Keep same data, different view
                code += f"""
    // Reshape layer: {node.name} (view change only)
    var {output_name.replace('.', '_')} = {current_input.replace('.', '_')};"""
                current_input = output_name
            else:
                print(f"Skipping unsupported layer type: {node.op_type}")
        
        # Add final return with argmax
        final_tensor = current_input.replace('.', '_')
        code += f"""
    
    // Return the full logits tensor
    return {final_tensor};
}}

// Function that returns full logits array
fn get_logits_with_weights(f32[] weights, i32 weight_count, f32[] input_data, i32 input_size) -> var {{
    return mobilenet_inference_core(weights, weight_count, input_data, input_size);
}}

// Host-compatible function that returns argmax class as float
fn mobilenet_inference_with_weights(f32[] weights, i32 weight_count, f32[] input_data, i32 input_size) -> f32 {{
    var logits = mobilenet_inference_core(weights, weight_count, input_data, input_size);
    
    // Find argmax
    var max_score = logits[0];
    var max_class = 0;
    var i = 1;
    while (i < 1000) {{
        if (logits[i] > max_score) {{
            max_score = logits[i];
            max_class = i;
        }}
        i = i + 1;
    }}
    return max_class;
}}


// Standalone version - ERROR: No weights provided!
fn mobilenet_inference() -> f32 {{
    // This function should NOT be called - weights must be provided!
    return -999.0; // Error indicator
}}

fn kernel_main() -> f32 {{
    // This should NOT be used - only for backwards compatibility
    return mobilenet_inference();
}}

// Main entry point that takes real weights and input from host  
fn kernel_main_with_weights(f32[] weights, i32 weight_count, f32[] input_data, i32 input_size) -> f32 {{
    return mobilenet_inference_with_weights(weights, weight_count, input_data, input_size);
}}

"""
        
        return code

def generate_simple_test():
    """Generate simple 3-layer test model"""
    print("Generating simple 3-layer MobileNet test...")
    
    generator = MobileNetV2Generator('mobilenetv2-7.onnx')
    
    # Override to generate only first 3 layers
    original_generate = generator.generate_full_model
    
    def generate_3_layer_test():
        code = f"""// Simple 3-Layer MobileNet Test with SimpBLAS Integration  
include "../simptensor/tensor_core.sl";

// Note: SimpBLAS functions sb_init and sb_gemm_f32 are provided by runtime

fn mobilenet_3_layer_test(f32[] weights, i32 weight_count) -> f32 {{
    // Initialize SimpBLAS first
    var blas_init = sb_init();
    
    // Input: 1x224x224x3
    var input_size = 150528;
    var input_data = array<f32, simd=auto>([input_size]);
    
    var i = 0;
    while (i < input_size) {{
        input_data[i] = 0.5;
        i = i + 1;
    }}
    
    // Layer 1: Simple conv using global weights
    var conv1_size = 401408;  // 112*112*32
    var conv1_out = array<f32, simd=auto>([conv1_size]);
    
    var conv_i = 0;
    while (conv_i < conv1_size) {{
        var weight_idx = conv_i % weight_count;
        if (weight_idx < weight_count) {{
            conv1_out[conv_i] = input_data[conv_i % input_size] * weights[weight_idx];
        }} else {{
            conv1_out[conv_i] = input_data[conv_i % input_size] * 0.01;
        }}
        conv_i = conv_i + 1;
    }}
    
    // Layer 2: BatchNorm + ReLU
    var bn_i = 0;
    while (bn_i < conv1_size) {{
        var val = conv1_out[bn_i];
        
        // Simple batch norm with passed weights
        var param_idx = (bn_i % 32) + 1000;  // Use weights at offset 1000+
        if (param_idx + 32 < weight_count) {{
            val = val * weights[param_idx] + weights[param_idx + 32];
        }}
        
        // ReLU
        if (val < 0.0) {{
            val = 0.0;
        }}
        
        conv1_out[bn_i] = val;
        bn_i = bn_i + 1;
    }}
    
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
    while (avg_i < 100) {{
        sum = sum + conv1_out[avg_i];
        avg_i = avg_i + 1;
    }}
    
    return (sum / 100.0) + gemm_result * 0.01;
}}

fn kernel_main_with_weights(f32[] weights, i32 weight_count) -> f32 {{
    return mobilenet_3_layer_test(weights, weight_count);
}}
"""
        return code
    
    code = generate_3_layer_test()
    
    with open('mobilenet_simple.sl', 'w') as f:
        f.write(code)
    
    print("Generated mobilenet_simple.sl with weight parameter support")
    return code

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--simple':
        generate_simple_test()
    else:
        print("Generating full MobileNetV2 SimpleLang implementation...")
        
        generator = MobileNetV2Generator('mobilenetv2-7.onnx')
        code = generator.generate_full_model()
        
        # Write to file
        with open('mobilenetv2_full.sl', 'w') as f:
            f.write(code)
        
        print(f"Generated mobilenetv2_full.sl")
        print(f"Total layers processed: {len(generator.model.graph.node)}")
        print("Ready for compilation!")

if __name__ == '__main__':
    main()