// Complete MobileNetV2 SimpleLang Implementation with SimpBLAS
include "../simptensor/tensor_core.sl";

// Note: SimpBLAS functions sb_init and sb_gemm_f32 are provided by runtime

// Helper function for im2col transformation
fn im2col_conv2d(f32[] input, i32 batch_size, i32 in_h, i32 in_w, i32 in_channels,
                 i32 kernel_h, i32 kernel_w, i32 stride_h, i32 stride_w,
                 i32 pad_h, i32 pad_w, i32 out_h, i32 out_w, f32[] col_buffer) -> void {
    var col_idx = 0;
    var out_y = 0;
    while (out_y < out_h) {
        var out_x = 0;
        while (out_x < out_w) {
            var ky = 0;
            while (ky < kernel_h) {
                var kx = 0;
                while (kx < kernel_w) {
                    var ch = 0;
                    while (ch < in_channels) {
                        var in_y = out_y * stride_h + ky - pad_h;
                        var in_x = out_x * stride_w + kx - pad_w;
                        
                        if (in_y >= 0) {
                            if (in_y < in_h) {
                                if (in_x >= 0) {
                                    if (in_x < in_w) {
                                        var in_offset = nhwc_offset(batch_size, in_h, in_w, in_channels, 0, in_y, in_x, ch);
                                        col_buffer[col_idx] = input[in_offset];
                                    } else {
                                        col_buffer[col_idx] = 0.0;
                                    }
                                } else {
                                    col_buffer[col_idx] = 0.0;
                                }
                            } else {
                                col_buffer[col_idx] = 0.0;
                            }
                        } else {
                            col_buffer[col_idx] = 0.0;
                        }
                        col_idx = col_idx + 1;
                        ch = ch + 1;
                    }
                    kx = kx + 1;
                }
                ky = ky + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
}

fn load_weights_from_binary() -> f32 {
    // Initialize SimpBLAS
    var init_result = sb_init();
    return 1.0;  // Success
}

// Core inference function that takes weights as parameters
fn mobilenet_inference_core(f32[] weights, i32 weight_count, f32[] input_data, i32 input_size) -> f32 {
    // Initialize SimpBLAS
    var init_result = sb_init();
    
    // Input tensor: 1x224x224x3 (NHWC)
    var input_size = 150528; // 1 * 224 * 224 * 3
    var data = array<f32, simd=auto>([input_size]);
    
    // Load input data from host
    var init_idx = 0;
    while (init_idx < input_size) {
        data[init_idx] = input_data[init_idx];
        init_idx = init_idx + 1;
    }

    // Conv layer: mobilenetv20_features_conv0_fwd
    // Weight shape: [32, 3, 3, 3], Groups: 1
    var weights_mobilenetv20_features_conv0_fwd_size = 864;
    var weights_mobilenetv20_features_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 0
    var wi = 0;
    while (wi < weights_mobilenetv20_features_conv0_fwd_size) {
        if (0 + wi < weight_count) {
            weights_mobilenetv20_features_conv0_fwd[wi] = weights[0 + wi];
        } else {
            weights_mobilenetv20_features_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_conv0_fwd
    var mobilenetv20_features_conv0_fwd_size = 401408;
    var mobilenetv20_features_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 32 x 27
    //   B (im2col):  27 x 12544 
    //   C (output):  32 x 12544
    
    // Allocate im2col buffer
    var col_size = 338688;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(data, 1, 224, 224, 3,
                  3, 3, 2, 2, 1, 1, 
                  112, 112, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([401408]);
    
    // GEMM call: C = A * B
    // A = weights [32 x 27]
    // B = col_buffer [27 x 12544]  
    // C = gemm_output [32 x 12544]
    sb_gemm_f32(32, 12544, 27,
                weights_mobilenetv20_features_conv0_fwd, 27,
                col_buffer, 12544, 
                gemm_output, 12544);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 112) {
        var out_x = 0;
        while (out_x < 112) {
            var out_ch = 0;
            while (out_ch < 32) {
                var gemm_idx = out_ch * 12544 + out_y * 112 + out_x;
                var out_offset = nhwc_offset(1, 112, 112, 32, 0, out_y, out_x, out_ch);
                mobilenetv20_features_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_batchnorm0_fwd = array<f32, simd=auto>([32]);
    var bn_bias_mobilenetv20_features_batchnorm0_fwd = array<f32, simd=auto>([32]);
    var bn_mean_mobilenetv20_features_batchnorm0_fwd = array<f32, simd=auto>([32]);
    var bn_var_mobilenetv20_features_batchnorm0_fwd = array<f32, simd=auto>([32]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 32) {
        if (864 + bni < weight_count) {
            bn_scale_mobilenetv20_features_batchnorm0_fwd[bni] = weights[864 + bni];
        }
        if (896 + bni < weight_count) {
            bn_bias_mobilenetv20_features_batchnorm0_fwd[bni] = weights[896 + bni];
        }
        if (928 + bni < weight_count) {
            bn_mean_mobilenetv20_features_batchnorm0_fwd[bni] = weights[928 + bni];
        }
        if (960 + bni < weight_count) {
            bn_var_mobilenetv20_features_batchnorm0_fwd[bni] = weights[960 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_batchnorm0_fwd = mobilenetv20_features_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 401408) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 32;
        var val = mobilenetv20_features_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_relu0_fwd
    var mobilenetv20_features_relu0_fwd = mobilenetv20_features_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 401408) {
        var val = mobilenetv20_features_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck0_conv0_fwd
    // Weight shape: [32, 32, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck0_conv0_fwd_size = 1024;
    var weights_mobilenetv20_features_linearbottleneck0_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck0_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 992
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck0_conv0_fwd_size) {
        if (992 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck0_conv0_fwd[wi] = weights[992 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck0_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck0_conv0_fwd
    var mobilenetv20_features_linearbottleneck0_conv0_fwd_size = 401408;
    var mobilenetv20_features_linearbottleneck0_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck0_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 32 x 32
    //   B (im2col):  32 x 12544 
    //   C (output):  32 x 12544
    
    // Allocate im2col buffer
    var col_size = 401408;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_relu0_fwd, 1, 112, 112, 32,
                  1, 1, 1, 1, 0, 0, 
                  112, 112, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([401408]);
    
    // GEMM call: C = A * B
    // A = weights [32 x 32]
    // B = col_buffer [32 x 12544]  
    // C = gemm_output [32 x 12544]
    sb_gemm_f32(32, 12544, 32,
                weights_mobilenetv20_features_linearbottleneck0_conv0_fwd, 32,
                col_buffer, 12544, 
                gemm_output, 12544);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 112) {
        var out_x = 0;
        while (out_x < 112) {
            var out_ch = 0;
            while (out_ch < 32) {
                var gemm_idx = out_ch * 12544 + out_y * 112 + out_x;
                var out_offset = nhwc_offset(1, 112, 112, 32, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck0_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck0_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd = array<f32, simd=auto>([32]);
    var bn_bias_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd = array<f32, simd=auto>([32]);
    var bn_mean_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd = array<f32, simd=auto>([32]);
    var bn_var_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd = array<f32, simd=auto>([32]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 32) {
        if (2016 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd[bni] = weights[2016 + bni];
        }
        if (2048 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd[bni] = weights[2048 + bni];
        }
        if (2080 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd[bni] = weights[2080 + bni];
        }
        if (2112 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd[bni] = weights[2112 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck0_batchnorm0_fwd = mobilenetv20_features_linearbottleneck0_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 401408) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 32;
        var val = mobilenetv20_features_linearbottleneck0_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck0_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck0_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck0_relu0_fwd
    var mobilenetv20_features_linearbottleneck0_relu0_fwd = mobilenetv20_features_linearbottleneck0_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 401408) {
        var val = mobilenetv20_features_linearbottleneck0_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck0_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck0_conv1_fwd
    // Weight shape: [32, 1, 3, 3], Groups: 32
    var weights_mobilenetv20_features_linearbottleneck0_conv1_fwd_size = 288;
    var weights_mobilenetv20_features_linearbottleneck0_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck0_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 2144
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck0_conv1_fwd_size) {
        if (2144 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck0_conv1_fwd[wi] = weights[2144 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck0_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck0_conv1_fwd
    var mobilenetv20_features_linearbottleneck0_conv1_fwd_size = 401408;
    var mobilenetv20_features_linearbottleneck0_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck0_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 32)
    var batch = 0;
    var out_y = 0;
    while (out_y < 112) {
        var out_x = 0;
        while (out_x < 112) {
            var out_ch = 0;
            while (out_ch < 32) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 112) {
                                if (in_x >= 0) {
                                    if (in_x < 112) {
                                        var in_offset = nhwc_offset(1, 112, 112, 32, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck0_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck0_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 112, 112, 32, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck0_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck0_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd = array<f32, simd=auto>([32]);
    var bn_bias_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd = array<f32, simd=auto>([32]);
    var bn_mean_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd = array<f32, simd=auto>([32]);
    var bn_var_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd = array<f32, simd=auto>([32]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 32) {
        if (2432 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd[bni] = weights[2432 + bni];
        }
        if (2464 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd[bni] = weights[2464 + bni];
        }
        if (2496 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd[bni] = weights[2496 + bni];
        }
        if (2528 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd[bni] = weights[2528 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck0_batchnorm1_fwd = mobilenetv20_features_linearbottleneck0_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 401408) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 32;
        var val = mobilenetv20_features_linearbottleneck0_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck0_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck0_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck0_relu1_fwd
    var mobilenetv20_features_linearbottleneck0_relu1_fwd = mobilenetv20_features_linearbottleneck0_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 401408) {
        var val = mobilenetv20_features_linearbottleneck0_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck0_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck0_conv2_fwd
    // Weight shape: [16, 32, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck0_conv2_fwd_size = 512;
    var weights_mobilenetv20_features_linearbottleneck0_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck0_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 2560
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck0_conv2_fwd_size) {
        if (2560 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck0_conv2_fwd[wi] = weights[2560 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck0_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck0_conv2_fwd
    var mobilenetv20_features_linearbottleneck0_conv2_fwd_size = 200704;
    var mobilenetv20_features_linearbottleneck0_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck0_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 16 x 32
    //   B (im2col):  32 x 12544 
    //   C (output):  16 x 12544
    
    // Allocate im2col buffer
    var col_size = 401408;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck0_relu1_fwd, 1, 112, 112, 32,
                  1, 1, 1, 1, 0, 0, 
                  112, 112, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([200704]);
    
    // GEMM call: C = A * B
    // A = weights [16 x 32]
    // B = col_buffer [32 x 12544]  
    // C = gemm_output [16 x 12544]
    sb_gemm_f32(16, 12544, 32,
                weights_mobilenetv20_features_linearbottleneck0_conv2_fwd, 32,
                col_buffer, 12544, 
                gemm_output, 12544);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 112) {
        var out_x = 0;
        while (out_x < 112) {
            var out_ch = 0;
            while (out_ch < 16) {
                var gemm_idx = out_ch * 12544 + out_y * 112 + out_x;
                var out_offset = nhwc_offset(1, 112, 112, 16, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck0_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck0_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd = array<f32, simd=auto>([16]);
    var bn_bias_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd = array<f32, simd=auto>([16]);
    var bn_mean_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd = array<f32, simd=auto>([16]);
    var bn_var_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd = array<f32, simd=auto>([16]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 16) {
        if (3072 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd[bni] = weights[3072 + bni];
        }
        if (3088 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd[bni] = weights[3088 + bni];
        }
        if (3104 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd[bni] = weights[3104 + bni];
        }
        if (3120 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd[bni] = weights[3120 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck0_batchnorm2_fwd = mobilenetv20_features_linearbottleneck0_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 200704) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 16;
        var val = mobilenetv20_features_linearbottleneck0_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck0_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck0_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck1_conv0_fwd
    // Weight shape: [96, 16, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck1_conv0_fwd_size = 1536;
    var weights_mobilenetv20_features_linearbottleneck1_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck1_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 3136
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck1_conv0_fwd_size) {
        if (3136 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck1_conv0_fwd[wi] = weights[3136 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck1_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck1_conv0_fwd
    var mobilenetv20_features_linearbottleneck1_conv0_fwd_size = 1204224;
    var mobilenetv20_features_linearbottleneck1_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck1_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 96 x 16
    //   B (im2col):  16 x 12544 
    //   C (output):  96 x 12544
    
    // Allocate im2col buffer
    var col_size = 200704;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck0_batchnorm2_fwd, 1, 112, 112, 16,
                  1, 1, 1, 1, 0, 0, 
                  112, 112, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([1204224]);
    
    // GEMM call: C = A * B
    // A = weights [96 x 16]
    // B = col_buffer [16 x 12544]  
    // C = gemm_output [96 x 12544]
    sb_gemm_f32(96, 12544, 16,
                weights_mobilenetv20_features_linearbottleneck1_conv0_fwd, 16,
                col_buffer, 12544, 
                gemm_output, 12544);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 112) {
        var out_x = 0;
        while (out_x < 112) {
            var out_ch = 0;
            while (out_ch < 96) {
                var gemm_idx = out_ch * 12544 + out_y * 112 + out_x;
                var out_offset = nhwc_offset(1, 112, 112, 96, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck1_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck1_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd = array<f32, simd=auto>([96]);
    var bn_bias_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd = array<f32, simd=auto>([96]);
    var bn_mean_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd = array<f32, simd=auto>([96]);
    var bn_var_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd = array<f32, simd=auto>([96]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 96) {
        if (4672 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd[bni] = weights[4672 + bni];
        }
        if (4768 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd[bni] = weights[4768 + bni];
        }
        if (4864 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd[bni] = weights[4864 + bni];
        }
        if (4960 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd[bni] = weights[4960 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck1_batchnorm0_fwd = mobilenetv20_features_linearbottleneck1_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 1204224) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 96;
        var val = mobilenetv20_features_linearbottleneck1_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck1_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck1_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck1_relu0_fwd
    var mobilenetv20_features_linearbottleneck1_relu0_fwd = mobilenetv20_features_linearbottleneck1_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 1204224) {
        var val = mobilenetv20_features_linearbottleneck1_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck1_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck1_conv1_fwd
    // Weight shape: [96, 1, 3, 3], Groups: 96
    var weights_mobilenetv20_features_linearbottleneck1_conv1_fwd_size = 864;
    var weights_mobilenetv20_features_linearbottleneck1_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck1_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 5056
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck1_conv1_fwd_size) {
        if (5056 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck1_conv1_fwd[wi] = weights[5056 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck1_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck1_conv1_fwd
    var mobilenetv20_features_linearbottleneck1_conv1_fwd_size = 301056;
    var mobilenetv20_features_linearbottleneck1_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck1_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 96)
    var batch = 0;
    var out_y = 0;
    while (out_y < 56) {
        var out_x = 0;
        while (out_x < 56) {
            var out_ch = 0;
            while (out_ch < 96) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 2 + ky - 1;
                        var in_x = out_x * 2 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 112) {
                                if (in_x >= 0) {
                                    if (in_x < 112) {
                                        var in_offset = nhwc_offset(1, 112, 112, 96, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck1_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck1_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 56, 56, 96, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck1_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck1_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd = array<f32, simd=auto>([96]);
    var bn_bias_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd = array<f32, simd=auto>([96]);
    var bn_mean_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd = array<f32, simd=auto>([96]);
    var bn_var_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd = array<f32, simd=auto>([96]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 96) {
        if (5920 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd[bni] = weights[5920 + bni];
        }
        if (6016 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd[bni] = weights[6016 + bni];
        }
        if (6112 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd[bni] = weights[6112 + bni];
        }
        if (6208 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd[bni] = weights[6208 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck1_batchnorm1_fwd = mobilenetv20_features_linearbottleneck1_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 301056) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 96;
        var val = mobilenetv20_features_linearbottleneck1_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck1_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck1_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck1_relu1_fwd
    var mobilenetv20_features_linearbottleneck1_relu1_fwd = mobilenetv20_features_linearbottleneck1_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 301056) {
        var val = mobilenetv20_features_linearbottleneck1_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck1_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck1_conv2_fwd
    // Weight shape: [24, 96, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck1_conv2_fwd_size = 2304;
    var weights_mobilenetv20_features_linearbottleneck1_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck1_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 6304
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck1_conv2_fwd_size) {
        if (6304 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck1_conv2_fwd[wi] = weights[6304 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck1_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck1_conv2_fwd
    var mobilenetv20_features_linearbottleneck1_conv2_fwd_size = 75264;
    var mobilenetv20_features_linearbottleneck1_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck1_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 24 x 96
    //   B (im2col):  96 x 3136 
    //   C (output):  24 x 3136
    
    // Allocate im2col buffer
    var col_size = 301056;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck1_relu1_fwd, 1, 56, 56, 96,
                  1, 1, 1, 1, 0, 0, 
                  56, 56, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([75264]);
    
    // GEMM call: C = A * B
    // A = weights [24 x 96]
    // B = col_buffer [96 x 3136]  
    // C = gemm_output [24 x 3136]
    sb_gemm_f32(24, 3136, 96,
                weights_mobilenetv20_features_linearbottleneck1_conv2_fwd, 96,
                col_buffer, 3136, 
                gemm_output, 3136);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 56) {
        var out_x = 0;
        while (out_x < 56) {
            var out_ch = 0;
            while (out_ch < 24) {
                var gemm_idx = out_ch * 3136 + out_y * 56 + out_x;
                var out_offset = nhwc_offset(1, 56, 56, 24, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck1_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck1_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd = array<f32, simd=auto>([24]);
    var bn_bias_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd = array<f32, simd=auto>([24]);
    var bn_mean_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd = array<f32, simd=auto>([24]);
    var bn_var_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd = array<f32, simd=auto>([24]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 24) {
        if (8608 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd[bni] = weights[8608 + bni];
        }
        if (8632 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd[bni] = weights[8632 + bni];
        }
        if (8656 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd[bni] = weights[8656 + bni];
        }
        if (8680 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd[bni] = weights[8680 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck1_batchnorm2_fwd = mobilenetv20_features_linearbottleneck1_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 75264) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 24;
        var val = mobilenetv20_features_linearbottleneck1_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck1_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck1_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck2_conv0_fwd
    // Weight shape: [144, 24, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck2_conv0_fwd_size = 3456;
    var weights_mobilenetv20_features_linearbottleneck2_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck2_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 8704
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck2_conv0_fwd_size) {
        if (8704 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck2_conv0_fwd[wi] = weights[8704 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck2_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck2_conv0_fwd
    var mobilenetv20_features_linearbottleneck2_conv0_fwd_size = 451584;
    var mobilenetv20_features_linearbottleneck2_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck2_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 144 x 24
    //   B (im2col):  24 x 3136 
    //   C (output):  144 x 3136
    
    // Allocate im2col buffer
    var col_size = 75264;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck1_batchnorm2_fwd, 1, 56, 56, 24,
                  1, 1, 1, 1, 0, 0, 
                  56, 56, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([451584]);
    
    // GEMM call: C = A * B
    // A = weights [144 x 24]
    // B = col_buffer [24 x 3136]  
    // C = gemm_output [144 x 3136]
    sb_gemm_f32(144, 3136, 24,
                weights_mobilenetv20_features_linearbottleneck2_conv0_fwd, 24,
                col_buffer, 3136, 
                gemm_output, 3136);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 56) {
        var out_x = 0;
        while (out_x < 56) {
            var out_ch = 0;
            while (out_ch < 144) {
                var gemm_idx = out_ch * 3136 + out_y * 56 + out_x;
                var out_offset = nhwc_offset(1, 56, 56, 144, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck2_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck2_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd = array<f32, simd=auto>([144]);
    var bn_bias_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd = array<f32, simd=auto>([144]);
    var bn_mean_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd = array<f32, simd=auto>([144]);
    var bn_var_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd = array<f32, simd=auto>([144]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 144) {
        if (12160 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd[bni] = weights[12160 + bni];
        }
        if (12304 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd[bni] = weights[12304 + bni];
        }
        if (12448 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd[bni] = weights[12448 + bni];
        }
        if (12592 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd[bni] = weights[12592 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck2_batchnorm0_fwd = mobilenetv20_features_linearbottleneck2_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 451584) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 144;
        var val = mobilenetv20_features_linearbottleneck2_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck2_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck2_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck2_relu0_fwd
    var mobilenetv20_features_linearbottleneck2_relu0_fwd = mobilenetv20_features_linearbottleneck2_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 451584) {
        var val = mobilenetv20_features_linearbottleneck2_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck2_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck2_conv1_fwd
    // Weight shape: [144, 1, 3, 3], Groups: 144
    var weights_mobilenetv20_features_linearbottleneck2_conv1_fwd_size = 1296;
    var weights_mobilenetv20_features_linearbottleneck2_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck2_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 12736
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck2_conv1_fwd_size) {
        if (12736 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck2_conv1_fwd[wi] = weights[12736 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck2_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck2_conv1_fwd
    var mobilenetv20_features_linearbottleneck2_conv1_fwd_size = 451584;
    var mobilenetv20_features_linearbottleneck2_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck2_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 144)
    var batch = 0;
    var out_y = 0;
    while (out_y < 56) {
        var out_x = 0;
        while (out_x < 56) {
            var out_ch = 0;
            while (out_ch < 144) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 56) {
                                if (in_x >= 0) {
                                    if (in_x < 56) {
                                        var in_offset = nhwc_offset(1, 56, 56, 144, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck2_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck2_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 56, 56, 144, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck2_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck2_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd = array<f32, simd=auto>([144]);
    var bn_bias_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd = array<f32, simd=auto>([144]);
    var bn_mean_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd = array<f32, simd=auto>([144]);
    var bn_var_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd = array<f32, simd=auto>([144]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 144) {
        if (14032 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd[bni] = weights[14032 + bni];
        }
        if (14176 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd[bni] = weights[14176 + bni];
        }
        if (14320 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd[bni] = weights[14320 + bni];
        }
        if (14464 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd[bni] = weights[14464 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck2_batchnorm1_fwd = mobilenetv20_features_linearbottleneck2_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 451584) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 144;
        var val = mobilenetv20_features_linearbottleneck2_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck2_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck2_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck2_relu1_fwd
    var mobilenetv20_features_linearbottleneck2_relu1_fwd = mobilenetv20_features_linearbottleneck2_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 451584) {
        var val = mobilenetv20_features_linearbottleneck2_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck2_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck2_conv2_fwd
    // Weight shape: [24, 144, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck2_conv2_fwd_size = 3456;
    var weights_mobilenetv20_features_linearbottleneck2_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck2_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 14608
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck2_conv2_fwd_size) {
        if (14608 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck2_conv2_fwd[wi] = weights[14608 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck2_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck2_conv2_fwd
    var mobilenetv20_features_linearbottleneck2_conv2_fwd_size = 75264;
    var mobilenetv20_features_linearbottleneck2_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck2_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 24 x 144
    //   B (im2col):  144 x 3136 
    //   C (output):  24 x 3136
    
    // Allocate im2col buffer
    var col_size = 451584;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck2_relu1_fwd, 1, 56, 56, 144,
                  1, 1, 1, 1, 0, 0, 
                  56, 56, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([75264]);
    
    // GEMM call: C = A * B
    // A = weights [24 x 144]
    // B = col_buffer [144 x 3136]  
    // C = gemm_output [24 x 3136]
    sb_gemm_f32(24, 3136, 144,
                weights_mobilenetv20_features_linearbottleneck2_conv2_fwd, 144,
                col_buffer, 3136, 
                gemm_output, 3136);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 56) {
        var out_x = 0;
        while (out_x < 56) {
            var out_ch = 0;
            while (out_ch < 24) {
                var gemm_idx = out_ch * 3136 + out_y * 56 + out_x;
                var out_offset = nhwc_offset(1, 56, 56, 24, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck2_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck2_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd = array<f32, simd=auto>([24]);
    var bn_bias_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd = array<f32, simd=auto>([24]);
    var bn_mean_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd = array<f32, simd=auto>([24]);
    var bn_var_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd = array<f32, simd=auto>([24]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 24) {
        if (18064 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd[bni] = weights[18064 + bni];
        }
        if (18088 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd[bni] = weights[18088 + bni];
        }
        if (18112 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd[bni] = weights[18112 + bni];
        }
        if (18136 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd[bni] = weights[18136 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck2_batchnorm2_fwd = mobilenetv20_features_linearbottleneck2_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 75264) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 24;
        var val = mobilenetv20_features_linearbottleneck2_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck2_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck2_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Add layer (residual): mobilenetv20_features_linearbottleneck2_elemwise_add0
    var mobilenetv20_features_linearbottleneck2_elemwise_add0_size = 75264;
    var mobilenetv20_features_linearbottleneck2_elemwise_add0 = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck2_elemwise_add0_size]);
    
    var add_idx = 0;
    while (add_idx < 75264) {
        mobilenetv20_features_linearbottleneck2_elemwise_add0[add_idx] = mobilenetv20_features_linearbottleneck2_batchnorm2_fwd[add_idx] + mobilenetv20_features_linearbottleneck1_batchnorm2_fwd[add_idx];
        add_idx = add_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck3_conv0_fwd
    // Weight shape: [144, 24, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck3_conv0_fwd_size = 3456;
    var weights_mobilenetv20_features_linearbottleneck3_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck3_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 18160
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck3_conv0_fwd_size) {
        if (18160 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck3_conv0_fwd[wi] = weights[18160 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck3_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck3_conv0_fwd
    var mobilenetv20_features_linearbottleneck3_conv0_fwd_size = 451584;
    var mobilenetv20_features_linearbottleneck3_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck3_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 144 x 24
    //   B (im2col):  24 x 3136 
    //   C (output):  144 x 3136
    
    // Allocate im2col buffer
    var col_size = 75264;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck2_elemwise_add0, 1, 56, 56, 24,
                  1, 1, 1, 1, 0, 0, 
                  56, 56, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([451584]);
    
    // GEMM call: C = A * B
    // A = weights [144 x 24]
    // B = col_buffer [24 x 3136]  
    // C = gemm_output [144 x 3136]
    sb_gemm_f32(144, 3136, 24,
                weights_mobilenetv20_features_linearbottleneck3_conv0_fwd, 24,
                col_buffer, 3136, 
                gemm_output, 3136);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 56) {
        var out_x = 0;
        while (out_x < 56) {
            var out_ch = 0;
            while (out_ch < 144) {
                var gemm_idx = out_ch * 3136 + out_y * 56 + out_x;
                var out_offset = nhwc_offset(1, 56, 56, 144, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck3_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck3_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd = array<f32, simd=auto>([144]);
    var bn_bias_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd = array<f32, simd=auto>([144]);
    var bn_mean_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd = array<f32, simd=auto>([144]);
    var bn_var_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd = array<f32, simd=auto>([144]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 144) {
        if (21616 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd[bni] = weights[21616 + bni];
        }
        if (21760 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd[bni] = weights[21760 + bni];
        }
        if (21904 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd[bni] = weights[21904 + bni];
        }
        if (22048 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd[bni] = weights[22048 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck3_batchnorm0_fwd = mobilenetv20_features_linearbottleneck3_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 451584) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 144;
        var val = mobilenetv20_features_linearbottleneck3_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck3_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck3_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck3_relu0_fwd
    var mobilenetv20_features_linearbottleneck3_relu0_fwd = mobilenetv20_features_linearbottleneck3_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 451584) {
        var val = mobilenetv20_features_linearbottleneck3_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck3_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck3_conv1_fwd
    // Weight shape: [144, 1, 3, 3], Groups: 144
    var weights_mobilenetv20_features_linearbottleneck3_conv1_fwd_size = 1296;
    var weights_mobilenetv20_features_linearbottleneck3_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck3_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 22192
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck3_conv1_fwd_size) {
        if (22192 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck3_conv1_fwd[wi] = weights[22192 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck3_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck3_conv1_fwd
    var mobilenetv20_features_linearbottleneck3_conv1_fwd_size = 112896;
    var mobilenetv20_features_linearbottleneck3_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck3_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 144)
    var batch = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 144) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 2 + ky - 1;
                        var in_x = out_x * 2 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 56) {
                                if (in_x >= 0) {
                                    if (in_x < 56) {
                                        var in_offset = nhwc_offset(1, 56, 56, 144, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck3_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck3_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 28, 28, 144, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck3_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck3_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd = array<f32, simd=auto>([144]);
    var bn_bias_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd = array<f32, simd=auto>([144]);
    var bn_mean_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd = array<f32, simd=auto>([144]);
    var bn_var_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd = array<f32, simd=auto>([144]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 144) {
        if (23488 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd[bni] = weights[23488 + bni];
        }
        if (23632 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd[bni] = weights[23632 + bni];
        }
        if (23776 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd[bni] = weights[23776 + bni];
        }
        if (23920 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd[bni] = weights[23920 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck3_batchnorm1_fwd = mobilenetv20_features_linearbottleneck3_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 112896) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 144;
        var val = mobilenetv20_features_linearbottleneck3_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck3_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck3_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck3_relu1_fwd
    var mobilenetv20_features_linearbottleneck3_relu1_fwd = mobilenetv20_features_linearbottleneck3_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 112896) {
        var val = mobilenetv20_features_linearbottleneck3_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck3_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck3_conv2_fwd
    // Weight shape: [32, 144, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck3_conv2_fwd_size = 4608;
    var weights_mobilenetv20_features_linearbottleneck3_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck3_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 24064
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck3_conv2_fwd_size) {
        if (24064 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck3_conv2_fwd[wi] = weights[24064 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck3_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck3_conv2_fwd
    var mobilenetv20_features_linearbottleneck3_conv2_fwd_size = 25088;
    var mobilenetv20_features_linearbottleneck3_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck3_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 32 x 144
    //   B (im2col):  144 x 784 
    //   C (output):  32 x 784
    
    // Allocate im2col buffer
    var col_size = 112896;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck3_relu1_fwd, 1, 28, 28, 144,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([25088]);
    
    // GEMM call: C = A * B
    // A = weights [32 x 144]
    // B = col_buffer [144 x 784]  
    // C = gemm_output [32 x 784]
    sb_gemm_f32(32, 784, 144,
                weights_mobilenetv20_features_linearbottleneck3_conv2_fwd, 144,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 32) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 32, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck3_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck3_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd = array<f32, simd=auto>([32]);
    var bn_bias_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd = array<f32, simd=auto>([32]);
    var bn_mean_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd = array<f32, simd=auto>([32]);
    var bn_var_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd = array<f32, simd=auto>([32]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 32) {
        if (28672 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd[bni] = weights[28672 + bni];
        }
        if (28704 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd[bni] = weights[28704 + bni];
        }
        if (28736 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd[bni] = weights[28736 + bni];
        }
        if (28768 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd[bni] = weights[28768 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck3_batchnorm2_fwd = mobilenetv20_features_linearbottleneck3_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 25088) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 32;
        var val = mobilenetv20_features_linearbottleneck3_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck3_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck3_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck4_conv0_fwd
    // Weight shape: [192, 32, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck4_conv0_fwd_size = 6144;
    var weights_mobilenetv20_features_linearbottleneck4_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck4_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 28800
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck4_conv0_fwd_size) {
        if (28800 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck4_conv0_fwd[wi] = weights[28800 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck4_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck4_conv0_fwd
    var mobilenetv20_features_linearbottleneck4_conv0_fwd_size = 150528;
    var mobilenetv20_features_linearbottleneck4_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck4_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 192 x 32
    //   B (im2col):  32 x 784 
    //   C (output):  192 x 784
    
    // Allocate im2col buffer
    var col_size = 25088;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck3_batchnorm2_fwd, 1, 28, 28, 32,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([150528]);
    
    // GEMM call: C = A * B
    // A = weights [192 x 32]
    // B = col_buffer [32 x 784]  
    // C = gemm_output [192 x 784]
    sb_gemm_f32(192, 784, 32,
                weights_mobilenetv20_features_linearbottleneck4_conv0_fwd, 32,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 192) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 192, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck4_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck4_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd = array<f32, simd=auto>([192]);
    var bn_bias_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd = array<f32, simd=auto>([192]);
    var bn_mean_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd = array<f32, simd=auto>([192]);
    var bn_var_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd = array<f32, simd=auto>([192]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 192) {
        if (34944 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd[bni] = weights[34944 + bni];
        }
        if (35136 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd[bni] = weights[35136 + bni];
        }
        if (35328 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd[bni] = weights[35328 + bni];
        }
        if (35520 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd[bni] = weights[35520 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck4_batchnorm0_fwd = mobilenetv20_features_linearbottleneck4_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 150528) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 192;
        var val = mobilenetv20_features_linearbottleneck4_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck4_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck4_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck4_relu0_fwd
    var mobilenetv20_features_linearbottleneck4_relu0_fwd = mobilenetv20_features_linearbottleneck4_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 150528) {
        var val = mobilenetv20_features_linearbottleneck4_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck4_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck4_conv1_fwd
    // Weight shape: [192, 1, 3, 3], Groups: 192
    var weights_mobilenetv20_features_linearbottleneck4_conv1_fwd_size = 1728;
    var weights_mobilenetv20_features_linearbottleneck4_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck4_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 35712
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck4_conv1_fwd_size) {
        if (35712 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck4_conv1_fwd[wi] = weights[35712 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck4_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck4_conv1_fwd
    var mobilenetv20_features_linearbottleneck4_conv1_fwd_size = 150528;
    var mobilenetv20_features_linearbottleneck4_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck4_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 192)
    var batch = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 192) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 28) {
                                if (in_x >= 0) {
                                    if (in_x < 28) {
                                        var in_offset = nhwc_offset(1, 28, 28, 192, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck4_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck4_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 28, 28, 192, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck4_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck4_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd = array<f32, simd=auto>([192]);
    var bn_bias_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd = array<f32, simd=auto>([192]);
    var bn_mean_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd = array<f32, simd=auto>([192]);
    var bn_var_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd = array<f32, simd=auto>([192]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 192) {
        if (37440 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd[bni] = weights[37440 + bni];
        }
        if (37632 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd[bni] = weights[37632 + bni];
        }
        if (37824 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd[bni] = weights[37824 + bni];
        }
        if (38016 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd[bni] = weights[38016 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck4_batchnorm1_fwd = mobilenetv20_features_linearbottleneck4_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 150528) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 192;
        var val = mobilenetv20_features_linearbottleneck4_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck4_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck4_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck4_relu1_fwd
    var mobilenetv20_features_linearbottleneck4_relu1_fwd = mobilenetv20_features_linearbottleneck4_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 150528) {
        var val = mobilenetv20_features_linearbottleneck4_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck4_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck4_conv2_fwd
    // Weight shape: [32, 192, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck4_conv2_fwd_size = 6144;
    var weights_mobilenetv20_features_linearbottleneck4_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck4_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 38208
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck4_conv2_fwd_size) {
        if (38208 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck4_conv2_fwd[wi] = weights[38208 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck4_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck4_conv2_fwd
    var mobilenetv20_features_linearbottleneck4_conv2_fwd_size = 25088;
    var mobilenetv20_features_linearbottleneck4_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck4_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 32 x 192
    //   B (im2col):  192 x 784 
    //   C (output):  32 x 784
    
    // Allocate im2col buffer
    var col_size = 150528;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck4_relu1_fwd, 1, 28, 28, 192,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([25088]);
    
    // GEMM call: C = A * B
    // A = weights [32 x 192]
    // B = col_buffer [192 x 784]  
    // C = gemm_output [32 x 784]
    sb_gemm_f32(32, 784, 192,
                weights_mobilenetv20_features_linearbottleneck4_conv2_fwd, 192,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 32) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 32, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck4_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck4_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd = array<f32, simd=auto>([32]);
    var bn_bias_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd = array<f32, simd=auto>([32]);
    var bn_mean_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd = array<f32, simd=auto>([32]);
    var bn_var_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd = array<f32, simd=auto>([32]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 32) {
        if (44352 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd[bni] = weights[44352 + bni];
        }
        if (44384 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd[bni] = weights[44384 + bni];
        }
        if (44416 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd[bni] = weights[44416 + bni];
        }
        if (44448 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd[bni] = weights[44448 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck4_batchnorm2_fwd = mobilenetv20_features_linearbottleneck4_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 25088) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 32;
        var val = mobilenetv20_features_linearbottleneck4_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck4_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck4_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Add layer (residual): mobilenetv20_features_linearbottleneck4_elemwise_add0
    var mobilenetv20_features_linearbottleneck4_elemwise_add0_size = 25088;
    var mobilenetv20_features_linearbottleneck4_elemwise_add0 = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck4_elemwise_add0_size]);
    
    var add_idx = 0;
    while (add_idx < 25088) {
        mobilenetv20_features_linearbottleneck4_elemwise_add0[add_idx] = mobilenetv20_features_linearbottleneck4_batchnorm2_fwd[add_idx] + mobilenetv20_features_linearbottleneck3_batchnorm2_fwd[add_idx];
        add_idx = add_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck5_conv0_fwd
    // Weight shape: [192, 32, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck5_conv0_fwd_size = 6144;
    var weights_mobilenetv20_features_linearbottleneck5_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck5_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 44480
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck5_conv0_fwd_size) {
        if (44480 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck5_conv0_fwd[wi] = weights[44480 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck5_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck5_conv0_fwd
    var mobilenetv20_features_linearbottleneck5_conv0_fwd_size = 150528;
    var mobilenetv20_features_linearbottleneck5_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck5_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 192 x 32
    //   B (im2col):  32 x 784 
    //   C (output):  192 x 784
    
    // Allocate im2col buffer
    var col_size = 25088;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck4_elemwise_add0, 1, 28, 28, 32,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([150528]);
    
    // GEMM call: C = A * B
    // A = weights [192 x 32]
    // B = col_buffer [32 x 784]  
    // C = gemm_output [192 x 784]
    sb_gemm_f32(192, 784, 32,
                weights_mobilenetv20_features_linearbottleneck5_conv0_fwd, 32,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 192) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 192, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck5_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck5_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd = array<f32, simd=auto>([192]);
    var bn_bias_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd = array<f32, simd=auto>([192]);
    var bn_mean_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd = array<f32, simd=auto>([192]);
    var bn_var_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd = array<f32, simd=auto>([192]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 192) {
        if (50624 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd[bni] = weights[50624 + bni];
        }
        if (50816 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd[bni] = weights[50816 + bni];
        }
        if (51008 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd[bni] = weights[51008 + bni];
        }
        if (51200 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd[bni] = weights[51200 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck5_batchnorm0_fwd = mobilenetv20_features_linearbottleneck5_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 150528) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 192;
        var val = mobilenetv20_features_linearbottleneck5_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck5_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck5_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck5_relu0_fwd
    var mobilenetv20_features_linearbottleneck5_relu0_fwd = mobilenetv20_features_linearbottleneck5_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 150528) {
        var val = mobilenetv20_features_linearbottleneck5_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck5_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck5_conv1_fwd
    // Weight shape: [192, 1, 3, 3], Groups: 192
    var weights_mobilenetv20_features_linearbottleneck5_conv1_fwd_size = 1728;
    var weights_mobilenetv20_features_linearbottleneck5_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck5_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 51392
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck5_conv1_fwd_size) {
        if (51392 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck5_conv1_fwd[wi] = weights[51392 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck5_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck5_conv1_fwd
    var mobilenetv20_features_linearbottleneck5_conv1_fwd_size = 150528;
    var mobilenetv20_features_linearbottleneck5_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck5_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 192)
    var batch = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 192) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 28) {
                                if (in_x >= 0) {
                                    if (in_x < 28) {
                                        var in_offset = nhwc_offset(1, 28, 28, 192, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck5_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck5_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 28, 28, 192, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck5_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck5_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd = array<f32, simd=auto>([192]);
    var bn_bias_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd = array<f32, simd=auto>([192]);
    var bn_mean_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd = array<f32, simd=auto>([192]);
    var bn_var_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd = array<f32, simd=auto>([192]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 192) {
        if (53120 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd[bni] = weights[53120 + bni];
        }
        if (53312 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd[bni] = weights[53312 + bni];
        }
        if (53504 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd[bni] = weights[53504 + bni];
        }
        if (53696 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd[bni] = weights[53696 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck5_batchnorm1_fwd = mobilenetv20_features_linearbottleneck5_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 150528) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 192;
        var val = mobilenetv20_features_linearbottleneck5_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck5_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck5_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck5_relu1_fwd
    var mobilenetv20_features_linearbottleneck5_relu1_fwd = mobilenetv20_features_linearbottleneck5_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 150528) {
        var val = mobilenetv20_features_linearbottleneck5_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck5_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck5_conv2_fwd
    // Weight shape: [32, 192, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck5_conv2_fwd_size = 6144;
    var weights_mobilenetv20_features_linearbottleneck5_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck5_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 53888
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck5_conv2_fwd_size) {
        if (53888 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck5_conv2_fwd[wi] = weights[53888 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck5_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck5_conv2_fwd
    var mobilenetv20_features_linearbottleneck5_conv2_fwd_size = 25088;
    var mobilenetv20_features_linearbottleneck5_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck5_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 32 x 192
    //   B (im2col):  192 x 784 
    //   C (output):  32 x 784
    
    // Allocate im2col buffer
    var col_size = 150528;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck5_relu1_fwd, 1, 28, 28, 192,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([25088]);
    
    // GEMM call: C = A * B
    // A = weights [32 x 192]
    // B = col_buffer [192 x 784]  
    // C = gemm_output [32 x 784]
    sb_gemm_f32(32, 784, 192,
                weights_mobilenetv20_features_linearbottleneck5_conv2_fwd, 192,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 32) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 32, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck5_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck5_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd = array<f32, simd=auto>([32]);
    var bn_bias_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd = array<f32, simd=auto>([32]);
    var bn_mean_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd = array<f32, simd=auto>([32]);
    var bn_var_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd = array<f32, simd=auto>([32]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 32) {
        if (60032 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd[bni] = weights[60032 + bni];
        }
        if (60064 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd[bni] = weights[60064 + bni];
        }
        if (60096 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd[bni] = weights[60096 + bni];
        }
        if (60128 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd[bni] = weights[60128 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck5_batchnorm2_fwd = mobilenetv20_features_linearbottleneck5_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 25088) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 32;
        var val = mobilenetv20_features_linearbottleneck5_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck5_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck5_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Add layer (residual): mobilenetv20_features_linearbottleneck5_elemwise_add0
    var mobilenetv20_features_linearbottleneck5_elemwise_add0_size = 25088;
    var mobilenetv20_features_linearbottleneck5_elemwise_add0 = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck5_elemwise_add0_size]);
    
    var add_idx = 0;
    while (add_idx < 25088) {
        mobilenetv20_features_linearbottleneck5_elemwise_add0[add_idx] = mobilenetv20_features_linearbottleneck5_batchnorm2_fwd[add_idx] + mobilenetv20_features_linearbottleneck4_elemwise_add0[add_idx];
        add_idx = add_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck6_conv0_fwd
    // Weight shape: [192, 32, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck6_conv0_fwd_size = 6144;
    var weights_mobilenetv20_features_linearbottleneck6_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck6_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 60160
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck6_conv0_fwd_size) {
        if (60160 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck6_conv0_fwd[wi] = weights[60160 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck6_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck6_conv0_fwd
    var mobilenetv20_features_linearbottleneck6_conv0_fwd_size = 150528;
    var mobilenetv20_features_linearbottleneck6_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck6_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 192 x 32
    //   B (im2col):  32 x 784 
    //   C (output):  192 x 784
    
    // Allocate im2col buffer
    var col_size = 25088;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck5_elemwise_add0, 1, 28, 28, 32,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([150528]);
    
    // GEMM call: C = A * B
    // A = weights [192 x 32]
    // B = col_buffer [32 x 784]  
    // C = gemm_output [192 x 784]
    sb_gemm_f32(192, 784, 32,
                weights_mobilenetv20_features_linearbottleneck6_conv0_fwd, 32,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 192) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 192, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck6_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck6_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd = array<f32, simd=auto>([192]);
    var bn_bias_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd = array<f32, simd=auto>([192]);
    var bn_mean_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd = array<f32, simd=auto>([192]);
    var bn_var_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd = array<f32, simd=auto>([192]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 192) {
        if (66304 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd[bni] = weights[66304 + bni];
        }
        if (66496 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd[bni] = weights[66496 + bni];
        }
        if (66688 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd[bni] = weights[66688 + bni];
        }
        if (66880 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd[bni] = weights[66880 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck6_batchnorm0_fwd = mobilenetv20_features_linearbottleneck6_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 150528) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 192;
        var val = mobilenetv20_features_linearbottleneck6_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck6_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck6_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck6_relu0_fwd
    var mobilenetv20_features_linearbottleneck6_relu0_fwd = mobilenetv20_features_linearbottleneck6_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 150528) {
        var val = mobilenetv20_features_linearbottleneck6_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck6_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck6_conv1_fwd
    // Weight shape: [192, 1, 3, 3], Groups: 192
    var weights_mobilenetv20_features_linearbottleneck6_conv1_fwd_size = 1728;
    var weights_mobilenetv20_features_linearbottleneck6_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck6_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 67072
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck6_conv1_fwd_size) {
        if (67072 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck6_conv1_fwd[wi] = weights[67072 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck6_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck6_conv1_fwd
    var mobilenetv20_features_linearbottleneck6_conv1_fwd_size = 150528;
    var mobilenetv20_features_linearbottleneck6_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck6_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 192)
    var batch = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 192) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 28) {
                                if (in_x >= 0) {
                                    if (in_x < 28) {
                                        var in_offset = nhwc_offset(1, 28, 28, 192, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck6_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck6_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 28, 28, 192, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck6_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck6_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd = array<f32, simd=auto>([192]);
    var bn_bias_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd = array<f32, simd=auto>([192]);
    var bn_mean_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd = array<f32, simd=auto>([192]);
    var bn_var_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd = array<f32, simd=auto>([192]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 192) {
        if (68800 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd[bni] = weights[68800 + bni];
        }
        if (68992 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd[bni] = weights[68992 + bni];
        }
        if (69184 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd[bni] = weights[69184 + bni];
        }
        if (69376 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd[bni] = weights[69376 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck6_batchnorm1_fwd = mobilenetv20_features_linearbottleneck6_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 150528) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 192;
        var val = mobilenetv20_features_linearbottleneck6_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck6_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck6_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck6_relu1_fwd
    var mobilenetv20_features_linearbottleneck6_relu1_fwd = mobilenetv20_features_linearbottleneck6_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 150528) {
        var val = mobilenetv20_features_linearbottleneck6_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck6_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck6_conv2_fwd
    // Weight shape: [64, 192, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck6_conv2_fwd_size = 12288;
    var weights_mobilenetv20_features_linearbottleneck6_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck6_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 69568
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck6_conv2_fwd_size) {
        if (69568 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck6_conv2_fwd[wi] = weights[69568 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck6_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck6_conv2_fwd
    var mobilenetv20_features_linearbottleneck6_conv2_fwd_size = 50176;
    var mobilenetv20_features_linearbottleneck6_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck6_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 64 x 192
    //   B (im2col):  192 x 784 
    //   C (output):  64 x 784
    
    // Allocate im2col buffer
    var col_size = 150528;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck6_relu1_fwd, 1, 28, 28, 192,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([50176]);
    
    // GEMM call: C = A * B
    // A = weights [64 x 192]
    // B = col_buffer [192 x 784]  
    // C = gemm_output [64 x 784]
    sb_gemm_f32(64, 784, 192,
                weights_mobilenetv20_features_linearbottleneck6_conv2_fwd, 192,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 64) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 64, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck6_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck6_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_bias_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_mean_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_var_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd = array<f32, simd=auto>([64]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 64) {
        if (81856 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd[bni] = weights[81856 + bni];
        }
        if (81920 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd[bni] = weights[81920 + bni];
        }
        if (81984 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd[bni] = weights[81984 + bni];
        }
        if (82048 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd[bni] = weights[82048 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck6_batchnorm2_fwd = mobilenetv20_features_linearbottleneck6_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 50176) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 64;
        var val = mobilenetv20_features_linearbottleneck6_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck6_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck6_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck7_conv0_fwd
    // Weight shape: [384, 64, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck7_conv0_fwd_size = 24576;
    var weights_mobilenetv20_features_linearbottleneck7_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck7_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 82112
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck7_conv0_fwd_size) {
        if (82112 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck7_conv0_fwd[wi] = weights[82112 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck7_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck7_conv0_fwd
    var mobilenetv20_features_linearbottleneck7_conv0_fwd_size = 301056;
    var mobilenetv20_features_linearbottleneck7_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck7_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 384 x 64
    //   B (im2col):  64 x 784 
    //   C (output):  384 x 784
    
    // Allocate im2col buffer
    var col_size = 50176;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck6_batchnorm2_fwd, 1, 28, 28, 64,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([301056]);
    
    // GEMM call: C = A * B
    // A = weights [384 x 64]
    // B = col_buffer [64 x 784]  
    // C = gemm_output [384 x 784]
    sb_gemm_f32(384, 784, 64,
                weights_mobilenetv20_features_linearbottleneck7_conv0_fwd, 64,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 384) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 384, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck7_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck7_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_bias_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_mean_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_var_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd = array<f32, simd=auto>([384]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 384) {
        if (106688 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd[bni] = weights[106688 + bni];
        }
        if (107072 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd[bni] = weights[107072 + bni];
        }
        if (107456 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd[bni] = weights[107456 + bni];
        }
        if (107840 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd[bni] = weights[107840 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck7_batchnorm0_fwd = mobilenetv20_features_linearbottleneck7_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 301056) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 384;
        var val = mobilenetv20_features_linearbottleneck7_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck7_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck7_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck7_relu0_fwd
    var mobilenetv20_features_linearbottleneck7_relu0_fwd = mobilenetv20_features_linearbottleneck7_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 301056) {
        var val = mobilenetv20_features_linearbottleneck7_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck7_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck7_conv1_fwd
    // Weight shape: [384, 1, 3, 3], Groups: 384
    var weights_mobilenetv20_features_linearbottleneck7_conv1_fwd_size = 3456;
    var weights_mobilenetv20_features_linearbottleneck7_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck7_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 108224
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck7_conv1_fwd_size) {
        if (108224 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck7_conv1_fwd[wi] = weights[108224 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck7_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck7_conv1_fwd
    var mobilenetv20_features_linearbottleneck7_conv1_fwd_size = 301056;
    var mobilenetv20_features_linearbottleneck7_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck7_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 384)
    var batch = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 384) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 28) {
                                if (in_x >= 0) {
                                    if (in_x < 28) {
                                        var in_offset = nhwc_offset(1, 28, 28, 384, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck7_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck7_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 28, 28, 384, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck7_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck7_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_bias_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_mean_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_var_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd = array<f32, simd=auto>([384]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 384) {
        if (111680 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd[bni] = weights[111680 + bni];
        }
        if (112064 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd[bni] = weights[112064 + bni];
        }
        if (112448 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd[bni] = weights[112448 + bni];
        }
        if (112832 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd[bni] = weights[112832 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck7_batchnorm1_fwd = mobilenetv20_features_linearbottleneck7_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 301056) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 384;
        var val = mobilenetv20_features_linearbottleneck7_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck7_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck7_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck7_relu1_fwd
    var mobilenetv20_features_linearbottleneck7_relu1_fwd = mobilenetv20_features_linearbottleneck7_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 301056) {
        var val = mobilenetv20_features_linearbottleneck7_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck7_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck7_conv2_fwd
    // Weight shape: [64, 384, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck7_conv2_fwd_size = 24576;
    var weights_mobilenetv20_features_linearbottleneck7_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck7_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 113216
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck7_conv2_fwd_size) {
        if (113216 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck7_conv2_fwd[wi] = weights[113216 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck7_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck7_conv2_fwd
    var mobilenetv20_features_linearbottleneck7_conv2_fwd_size = 50176;
    var mobilenetv20_features_linearbottleneck7_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck7_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 64 x 384
    //   B (im2col):  384 x 784 
    //   C (output):  64 x 784
    
    // Allocate im2col buffer
    var col_size = 301056;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck7_relu1_fwd, 1, 28, 28, 384,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([50176]);
    
    // GEMM call: C = A * B
    // A = weights [64 x 384]
    // B = col_buffer [384 x 784]  
    // C = gemm_output [64 x 784]
    sb_gemm_f32(64, 784, 384,
                weights_mobilenetv20_features_linearbottleneck7_conv2_fwd, 384,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 64) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 64, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck7_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck7_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_bias_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_mean_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_var_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd = array<f32, simd=auto>([64]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 64) {
        if (137792 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd[bni] = weights[137792 + bni];
        }
        if (137856 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd[bni] = weights[137856 + bni];
        }
        if (137920 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd[bni] = weights[137920 + bni];
        }
        if (137984 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd[bni] = weights[137984 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck7_batchnorm2_fwd = mobilenetv20_features_linearbottleneck7_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 50176) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 64;
        var val = mobilenetv20_features_linearbottleneck7_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck7_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck7_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Add layer (residual): mobilenetv20_features_linearbottleneck7_elemwise_add0
    var mobilenetv20_features_linearbottleneck7_elemwise_add0_size = 50176;
    var mobilenetv20_features_linearbottleneck7_elemwise_add0 = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck7_elemwise_add0_size]);
    
    var add_idx = 0;
    while (add_idx < 50176) {
        mobilenetv20_features_linearbottleneck7_elemwise_add0[add_idx] = mobilenetv20_features_linearbottleneck7_batchnorm2_fwd[add_idx] + mobilenetv20_features_linearbottleneck6_batchnorm2_fwd[add_idx];
        add_idx = add_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck8_conv0_fwd
    // Weight shape: [384, 64, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck8_conv0_fwd_size = 24576;
    var weights_mobilenetv20_features_linearbottleneck8_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck8_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 138048
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck8_conv0_fwd_size) {
        if (138048 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck8_conv0_fwd[wi] = weights[138048 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck8_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck8_conv0_fwd
    var mobilenetv20_features_linearbottleneck8_conv0_fwd_size = 301056;
    var mobilenetv20_features_linearbottleneck8_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck8_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 384 x 64
    //   B (im2col):  64 x 784 
    //   C (output):  384 x 784
    
    // Allocate im2col buffer
    var col_size = 50176;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck7_elemwise_add0, 1, 28, 28, 64,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([301056]);
    
    // GEMM call: C = A * B
    // A = weights [384 x 64]
    // B = col_buffer [64 x 784]  
    // C = gemm_output [384 x 784]
    sb_gemm_f32(384, 784, 64,
                weights_mobilenetv20_features_linearbottleneck8_conv0_fwd, 64,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 384) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 384, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck8_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck8_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_bias_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_mean_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_var_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd = array<f32, simd=auto>([384]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 384) {
        if (162624 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd[bni] = weights[162624 + bni];
        }
        if (163008 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd[bni] = weights[163008 + bni];
        }
        if (163392 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd[bni] = weights[163392 + bni];
        }
        if (163776 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd[bni] = weights[163776 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck8_batchnorm0_fwd = mobilenetv20_features_linearbottleneck8_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 301056) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 384;
        var val = mobilenetv20_features_linearbottleneck8_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck8_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck8_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck8_relu0_fwd
    var mobilenetv20_features_linearbottleneck8_relu0_fwd = mobilenetv20_features_linearbottleneck8_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 301056) {
        var val = mobilenetv20_features_linearbottleneck8_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck8_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck8_conv1_fwd
    // Weight shape: [384, 1, 3, 3], Groups: 384
    var weights_mobilenetv20_features_linearbottleneck8_conv1_fwd_size = 3456;
    var weights_mobilenetv20_features_linearbottleneck8_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck8_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 164160
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck8_conv1_fwd_size) {
        if (164160 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck8_conv1_fwd[wi] = weights[164160 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck8_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck8_conv1_fwd
    var mobilenetv20_features_linearbottleneck8_conv1_fwd_size = 301056;
    var mobilenetv20_features_linearbottleneck8_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck8_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 384)
    var batch = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 384) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 28) {
                                if (in_x >= 0) {
                                    if (in_x < 28) {
                                        var in_offset = nhwc_offset(1, 28, 28, 384, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck8_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck8_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 28, 28, 384, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck8_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck8_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_bias_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_mean_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_var_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd = array<f32, simd=auto>([384]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 384) {
        if (167616 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd[bni] = weights[167616 + bni];
        }
        if (168000 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd[bni] = weights[168000 + bni];
        }
        if (168384 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd[bni] = weights[168384 + bni];
        }
        if (168768 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd[bni] = weights[168768 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck8_batchnorm1_fwd = mobilenetv20_features_linearbottleneck8_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 301056) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 384;
        var val = mobilenetv20_features_linearbottleneck8_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck8_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck8_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck8_relu1_fwd
    var mobilenetv20_features_linearbottleneck8_relu1_fwd = mobilenetv20_features_linearbottleneck8_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 301056) {
        var val = mobilenetv20_features_linearbottleneck8_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck8_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck8_conv2_fwd
    // Weight shape: [64, 384, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck8_conv2_fwd_size = 24576;
    var weights_mobilenetv20_features_linearbottleneck8_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck8_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 169152
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck8_conv2_fwd_size) {
        if (169152 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck8_conv2_fwd[wi] = weights[169152 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck8_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck8_conv2_fwd
    var mobilenetv20_features_linearbottleneck8_conv2_fwd_size = 50176;
    var mobilenetv20_features_linearbottleneck8_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck8_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 64 x 384
    //   B (im2col):  384 x 784 
    //   C (output):  64 x 784
    
    // Allocate im2col buffer
    var col_size = 301056;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck8_relu1_fwd, 1, 28, 28, 384,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([50176]);
    
    // GEMM call: C = A * B
    // A = weights [64 x 384]
    // B = col_buffer [384 x 784]  
    // C = gemm_output [64 x 784]
    sb_gemm_f32(64, 784, 384,
                weights_mobilenetv20_features_linearbottleneck8_conv2_fwd, 384,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 64) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 64, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck8_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck8_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_bias_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_mean_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_var_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd = array<f32, simd=auto>([64]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 64) {
        if (193728 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd[bni] = weights[193728 + bni];
        }
        if (193792 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd[bni] = weights[193792 + bni];
        }
        if (193856 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd[bni] = weights[193856 + bni];
        }
        if (193920 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd[bni] = weights[193920 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck8_batchnorm2_fwd = mobilenetv20_features_linearbottleneck8_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 50176) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 64;
        var val = mobilenetv20_features_linearbottleneck8_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck8_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck8_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Add layer (residual): mobilenetv20_features_linearbottleneck8_elemwise_add0
    var mobilenetv20_features_linearbottleneck8_elemwise_add0_size = 50176;
    var mobilenetv20_features_linearbottleneck8_elemwise_add0 = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck8_elemwise_add0_size]);
    
    var add_idx = 0;
    while (add_idx < 50176) {
        mobilenetv20_features_linearbottleneck8_elemwise_add0[add_idx] = mobilenetv20_features_linearbottleneck8_batchnorm2_fwd[add_idx] + mobilenetv20_features_linearbottleneck7_elemwise_add0[add_idx];
        add_idx = add_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck9_conv0_fwd
    // Weight shape: [384, 64, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck9_conv0_fwd_size = 24576;
    var weights_mobilenetv20_features_linearbottleneck9_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck9_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 193984
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck9_conv0_fwd_size) {
        if (193984 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck9_conv0_fwd[wi] = weights[193984 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck9_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck9_conv0_fwd
    var mobilenetv20_features_linearbottleneck9_conv0_fwd_size = 301056;
    var mobilenetv20_features_linearbottleneck9_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck9_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 384 x 64
    //   B (im2col):  64 x 784 
    //   C (output):  384 x 784
    
    // Allocate im2col buffer
    var col_size = 50176;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck8_elemwise_add0, 1, 28, 28, 64,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([301056]);
    
    // GEMM call: C = A * B
    // A = weights [384 x 64]
    // B = col_buffer [64 x 784]  
    // C = gemm_output [384 x 784]
    sb_gemm_f32(384, 784, 64,
                weights_mobilenetv20_features_linearbottleneck9_conv0_fwd, 64,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 384) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 384, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck9_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck9_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_bias_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_mean_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_var_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd = array<f32, simd=auto>([384]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 384) {
        if (218560 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd[bni] = weights[218560 + bni];
        }
        if (218944 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd[bni] = weights[218944 + bni];
        }
        if (219328 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd[bni] = weights[219328 + bni];
        }
        if (219712 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd[bni] = weights[219712 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck9_batchnorm0_fwd = mobilenetv20_features_linearbottleneck9_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 301056) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 384;
        var val = mobilenetv20_features_linearbottleneck9_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck9_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck9_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck9_relu0_fwd
    var mobilenetv20_features_linearbottleneck9_relu0_fwd = mobilenetv20_features_linearbottleneck9_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 301056) {
        var val = mobilenetv20_features_linearbottleneck9_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck9_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck9_conv1_fwd
    // Weight shape: [384, 1, 3, 3], Groups: 384
    var weights_mobilenetv20_features_linearbottleneck9_conv1_fwd_size = 3456;
    var weights_mobilenetv20_features_linearbottleneck9_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck9_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 220096
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck9_conv1_fwd_size) {
        if (220096 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck9_conv1_fwd[wi] = weights[220096 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck9_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck9_conv1_fwd
    var mobilenetv20_features_linearbottleneck9_conv1_fwd_size = 301056;
    var mobilenetv20_features_linearbottleneck9_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck9_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 384)
    var batch = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 384) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 28) {
                                if (in_x >= 0) {
                                    if (in_x < 28) {
                                        var in_offset = nhwc_offset(1, 28, 28, 384, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck9_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck9_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 28, 28, 384, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck9_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck9_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_bias_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_mean_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_var_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd = array<f32, simd=auto>([384]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 384) {
        if (223552 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd[bni] = weights[223552 + bni];
        }
        if (223936 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd[bni] = weights[223936 + bni];
        }
        if (224320 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd[bni] = weights[224320 + bni];
        }
        if (224704 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd[bni] = weights[224704 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck9_batchnorm1_fwd = mobilenetv20_features_linearbottleneck9_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 301056) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 384;
        var val = mobilenetv20_features_linearbottleneck9_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck9_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck9_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck9_relu1_fwd
    var mobilenetv20_features_linearbottleneck9_relu1_fwd = mobilenetv20_features_linearbottleneck9_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 301056) {
        var val = mobilenetv20_features_linearbottleneck9_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck9_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck9_conv2_fwd
    // Weight shape: [64, 384, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck9_conv2_fwd_size = 24576;
    var weights_mobilenetv20_features_linearbottleneck9_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck9_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 225088
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck9_conv2_fwd_size) {
        if (225088 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck9_conv2_fwd[wi] = weights[225088 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck9_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck9_conv2_fwd
    var mobilenetv20_features_linearbottleneck9_conv2_fwd_size = 50176;
    var mobilenetv20_features_linearbottleneck9_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck9_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 64 x 384
    //   B (im2col):  384 x 784 
    //   C (output):  64 x 784
    
    // Allocate im2col buffer
    var col_size = 301056;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck9_relu1_fwd, 1, 28, 28, 384,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([50176]);
    
    // GEMM call: C = A * B
    // A = weights [64 x 384]
    // B = col_buffer [384 x 784]  
    // C = gemm_output [64 x 784]
    sb_gemm_f32(64, 784, 384,
                weights_mobilenetv20_features_linearbottleneck9_conv2_fwd, 384,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 64) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 64, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck9_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck9_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_bias_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_mean_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd = array<f32, simd=auto>([64]);
    var bn_var_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd = array<f32, simd=auto>([64]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 64) {
        if (249664 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd[bni] = weights[249664 + bni];
        }
        if (249728 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd[bni] = weights[249728 + bni];
        }
        if (249792 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd[bni] = weights[249792 + bni];
        }
        if (249856 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd[bni] = weights[249856 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck9_batchnorm2_fwd = mobilenetv20_features_linearbottleneck9_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 50176) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 64;
        var val = mobilenetv20_features_linearbottleneck9_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck9_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck9_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Add layer (residual): mobilenetv20_features_linearbottleneck9_elemwise_add0
    var mobilenetv20_features_linearbottleneck9_elemwise_add0_size = 50176;
    var mobilenetv20_features_linearbottleneck9_elemwise_add0 = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck9_elemwise_add0_size]);
    
    var add_idx = 0;
    while (add_idx < 50176) {
        mobilenetv20_features_linearbottleneck9_elemwise_add0[add_idx] = mobilenetv20_features_linearbottleneck9_batchnorm2_fwd[add_idx] + mobilenetv20_features_linearbottleneck8_elemwise_add0[add_idx];
        add_idx = add_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck10_conv0_fwd
    // Weight shape: [384, 64, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck10_conv0_fwd_size = 24576;
    var weights_mobilenetv20_features_linearbottleneck10_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck10_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 249920
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck10_conv0_fwd_size) {
        if (249920 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck10_conv0_fwd[wi] = weights[249920 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck10_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck10_conv0_fwd
    var mobilenetv20_features_linearbottleneck10_conv0_fwd_size = 301056;
    var mobilenetv20_features_linearbottleneck10_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck10_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 384 x 64
    //   B (im2col):  64 x 784 
    //   C (output):  384 x 784
    
    // Allocate im2col buffer
    var col_size = 50176;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck9_elemwise_add0, 1, 28, 28, 64,
                  1, 1, 1, 1, 0, 0, 
                  28, 28, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([301056]);
    
    // GEMM call: C = A * B
    // A = weights [384 x 64]
    // B = col_buffer [64 x 784]  
    // C = gemm_output [384 x 784]
    sb_gemm_f32(384, 784, 64,
                weights_mobilenetv20_features_linearbottleneck10_conv0_fwd, 64,
                col_buffer, 784, 
                gemm_output, 784);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 28) {
        var out_x = 0;
        while (out_x < 28) {
            var out_ch = 0;
            while (out_ch < 384) {
                var gemm_idx = out_ch * 784 + out_y * 28 + out_x;
                var out_offset = nhwc_offset(1, 28, 28, 384, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck10_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck10_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_bias_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_mean_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd = array<f32, simd=auto>([384]);
    var bn_var_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd = array<f32, simd=auto>([384]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 384) {
        if (274496 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd[bni] = weights[274496 + bni];
        }
        if (274880 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd[bni] = weights[274880 + bni];
        }
        if (275264 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd[bni] = weights[275264 + bni];
        }
        if (275648 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd[bni] = weights[275648 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck10_batchnorm0_fwd = mobilenetv20_features_linearbottleneck10_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 301056) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 384;
        var val = mobilenetv20_features_linearbottleneck10_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck10_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck10_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck10_relu0_fwd
    var mobilenetv20_features_linearbottleneck10_relu0_fwd = mobilenetv20_features_linearbottleneck10_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 301056) {
        var val = mobilenetv20_features_linearbottleneck10_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck10_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck10_conv1_fwd
    // Weight shape: [384, 1, 3, 3], Groups: 384
    var weights_mobilenetv20_features_linearbottleneck10_conv1_fwd_size = 3456;
    var weights_mobilenetv20_features_linearbottleneck10_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck10_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 276032
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck10_conv1_fwd_size) {
        if (276032 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck10_conv1_fwd[wi] = weights[276032 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck10_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck10_conv1_fwd
    var mobilenetv20_features_linearbottleneck10_conv1_fwd_size = 75264;
    var mobilenetv20_features_linearbottleneck10_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck10_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 384)
    var batch = 0;
    var out_y = 0;
    while (out_y < 14) {
        var out_x = 0;
        while (out_x < 14) {
            var out_ch = 0;
            while (out_ch < 384) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 2 + ky - 1;
                        var in_x = out_x * 2 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 28) {
                                if (in_x >= 0) {
                                    if (in_x < 28) {
                                        var in_offset = nhwc_offset(1, 28, 28, 384, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck10_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck10_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 14, 14, 384, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck10_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck10_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_bias_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_mean_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd = array<f32, simd=auto>([384]);
    var bn_var_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd = array<f32, simd=auto>([384]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 384) {
        if (279488 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd[bni] = weights[279488 + bni];
        }
        if (279872 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd[bni] = weights[279872 + bni];
        }
        if (280256 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd[bni] = weights[280256 + bni];
        }
        if (280640 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd[bni] = weights[280640 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck10_batchnorm1_fwd = mobilenetv20_features_linearbottleneck10_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 75264) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 384;
        var val = mobilenetv20_features_linearbottleneck10_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck10_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck10_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck10_relu1_fwd
    var mobilenetv20_features_linearbottleneck10_relu1_fwd = mobilenetv20_features_linearbottleneck10_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 75264) {
        var val = mobilenetv20_features_linearbottleneck10_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck10_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck10_conv2_fwd
    // Weight shape: [96, 384, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck10_conv2_fwd_size = 36864;
    var weights_mobilenetv20_features_linearbottleneck10_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck10_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 281024
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck10_conv2_fwd_size) {
        if (281024 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck10_conv2_fwd[wi] = weights[281024 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck10_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck10_conv2_fwd
    var mobilenetv20_features_linearbottleneck10_conv2_fwd_size = 18816;
    var mobilenetv20_features_linearbottleneck10_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck10_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 96 x 384
    //   B (im2col):  384 x 196 
    //   C (output):  96 x 196
    
    // Allocate im2col buffer
    var col_size = 75264;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck10_relu1_fwd, 1, 14, 14, 384,
                  1, 1, 1, 1, 0, 0, 
                  14, 14, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([18816]);
    
    // GEMM call: C = A * B
    // A = weights [96 x 384]
    // B = col_buffer [384 x 196]  
    // C = gemm_output [96 x 196]
    sb_gemm_f32(96, 196, 384,
                weights_mobilenetv20_features_linearbottleneck10_conv2_fwd, 384,
                col_buffer, 196, 
                gemm_output, 196);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 14) {
        var out_x = 0;
        while (out_x < 14) {
            var out_ch = 0;
            while (out_ch < 96) {
                var gemm_idx = out_ch * 196 + out_y * 14 + out_x;
                var out_offset = nhwc_offset(1, 14, 14, 96, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck10_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck10_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd = array<f32, simd=auto>([96]);
    var bn_bias_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd = array<f32, simd=auto>([96]);
    var bn_mean_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd = array<f32, simd=auto>([96]);
    var bn_var_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd = array<f32, simd=auto>([96]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 96) {
        if (317888 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd[bni] = weights[317888 + bni];
        }
        if (317984 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd[bni] = weights[317984 + bni];
        }
        if (318080 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd[bni] = weights[318080 + bni];
        }
        if (318176 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd[bni] = weights[318176 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck10_batchnorm2_fwd = mobilenetv20_features_linearbottleneck10_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 18816) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 96;
        var val = mobilenetv20_features_linearbottleneck10_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck10_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck10_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck11_conv0_fwd
    // Weight shape: [576, 96, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck11_conv0_fwd_size = 55296;
    var weights_mobilenetv20_features_linearbottleneck11_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck11_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 318272
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck11_conv0_fwd_size) {
        if (318272 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck11_conv0_fwd[wi] = weights[318272 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck11_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck11_conv0_fwd
    var mobilenetv20_features_linearbottleneck11_conv0_fwd_size = 112896;
    var mobilenetv20_features_linearbottleneck11_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck11_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 576 x 96
    //   B (im2col):  96 x 196 
    //   C (output):  576 x 196
    
    // Allocate im2col buffer
    var col_size = 18816;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck10_batchnorm2_fwd, 1, 14, 14, 96,
                  1, 1, 1, 1, 0, 0, 
                  14, 14, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([112896]);
    
    // GEMM call: C = A * B
    // A = weights [576 x 96]
    // B = col_buffer [96 x 196]  
    // C = gemm_output [576 x 196]
    sb_gemm_f32(576, 196, 96,
                weights_mobilenetv20_features_linearbottleneck11_conv0_fwd, 96,
                col_buffer, 196, 
                gemm_output, 196);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 14) {
        var out_x = 0;
        while (out_x < 14) {
            var out_ch = 0;
            while (out_ch < 576) {
                var gemm_idx = out_ch * 196 + out_y * 14 + out_x;
                var out_offset = nhwc_offset(1, 14, 14, 576, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck11_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck11_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd = array<f32, simd=auto>([576]);
    var bn_bias_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd = array<f32, simd=auto>([576]);
    var bn_mean_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd = array<f32, simd=auto>([576]);
    var bn_var_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd = array<f32, simd=auto>([576]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 576) {
        if (373568 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd[bni] = weights[373568 + bni];
        }
        if (374144 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd[bni] = weights[374144 + bni];
        }
        if (374720 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd[bni] = weights[374720 + bni];
        }
        if (375296 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd[bni] = weights[375296 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck11_batchnorm0_fwd = mobilenetv20_features_linearbottleneck11_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 112896) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 576;
        var val = mobilenetv20_features_linearbottleneck11_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck11_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck11_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck11_relu0_fwd
    var mobilenetv20_features_linearbottleneck11_relu0_fwd = mobilenetv20_features_linearbottleneck11_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 112896) {
        var val = mobilenetv20_features_linearbottleneck11_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck11_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck11_conv1_fwd
    // Weight shape: [576, 1, 3, 3], Groups: 576
    var weights_mobilenetv20_features_linearbottleneck11_conv1_fwd_size = 5184;
    var weights_mobilenetv20_features_linearbottleneck11_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck11_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 375872
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck11_conv1_fwd_size) {
        if (375872 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck11_conv1_fwd[wi] = weights[375872 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck11_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck11_conv1_fwd
    var mobilenetv20_features_linearbottleneck11_conv1_fwd_size = 112896;
    var mobilenetv20_features_linearbottleneck11_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck11_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 576)
    var batch = 0;
    var out_y = 0;
    while (out_y < 14) {
        var out_x = 0;
        while (out_x < 14) {
            var out_ch = 0;
            while (out_ch < 576) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 14) {
                                if (in_x >= 0) {
                                    if (in_x < 14) {
                                        var in_offset = nhwc_offset(1, 14, 14, 576, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck11_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck11_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 14, 14, 576, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck11_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck11_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd = array<f32, simd=auto>([576]);
    var bn_bias_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd = array<f32, simd=auto>([576]);
    var bn_mean_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd = array<f32, simd=auto>([576]);
    var bn_var_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd = array<f32, simd=auto>([576]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 576) {
        if (381056 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd[bni] = weights[381056 + bni];
        }
        if (381632 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd[bni] = weights[381632 + bni];
        }
        if (382208 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd[bni] = weights[382208 + bni];
        }
        if (382784 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd[bni] = weights[382784 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck11_batchnorm1_fwd = mobilenetv20_features_linearbottleneck11_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 112896) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 576;
        var val = mobilenetv20_features_linearbottleneck11_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck11_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck11_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck11_relu1_fwd
    var mobilenetv20_features_linearbottleneck11_relu1_fwd = mobilenetv20_features_linearbottleneck11_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 112896) {
        var val = mobilenetv20_features_linearbottleneck11_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck11_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck11_conv2_fwd
    // Weight shape: [96, 576, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck11_conv2_fwd_size = 55296;
    var weights_mobilenetv20_features_linearbottleneck11_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck11_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 383360
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck11_conv2_fwd_size) {
        if (383360 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck11_conv2_fwd[wi] = weights[383360 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck11_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck11_conv2_fwd
    var mobilenetv20_features_linearbottleneck11_conv2_fwd_size = 18816;
    var mobilenetv20_features_linearbottleneck11_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck11_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 96 x 576
    //   B (im2col):  576 x 196 
    //   C (output):  96 x 196
    
    // Allocate im2col buffer
    var col_size = 112896;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck11_relu1_fwd, 1, 14, 14, 576,
                  1, 1, 1, 1, 0, 0, 
                  14, 14, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([18816]);
    
    // GEMM call: C = A * B
    // A = weights [96 x 576]
    // B = col_buffer [576 x 196]  
    // C = gemm_output [96 x 196]
    sb_gemm_f32(96, 196, 576,
                weights_mobilenetv20_features_linearbottleneck11_conv2_fwd, 576,
                col_buffer, 196, 
                gemm_output, 196);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 14) {
        var out_x = 0;
        while (out_x < 14) {
            var out_ch = 0;
            while (out_ch < 96) {
                var gemm_idx = out_ch * 196 + out_y * 14 + out_x;
                var out_offset = nhwc_offset(1, 14, 14, 96, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck11_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck11_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd = array<f32, simd=auto>([96]);
    var bn_bias_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd = array<f32, simd=auto>([96]);
    var bn_mean_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd = array<f32, simd=auto>([96]);
    var bn_var_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd = array<f32, simd=auto>([96]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 96) {
        if (438656 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd[bni] = weights[438656 + bni];
        }
        if (438752 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd[bni] = weights[438752 + bni];
        }
        if (438848 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd[bni] = weights[438848 + bni];
        }
        if (438944 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd[bni] = weights[438944 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck11_batchnorm2_fwd = mobilenetv20_features_linearbottleneck11_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 18816) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 96;
        var val = mobilenetv20_features_linearbottleneck11_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck11_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck11_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Add layer (residual): mobilenetv20_features_linearbottleneck11_elemwise_add0
    var mobilenetv20_features_linearbottleneck11_elemwise_add0_size = 18816;
    var mobilenetv20_features_linearbottleneck11_elemwise_add0 = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck11_elemwise_add0_size]);
    
    var add_idx = 0;
    while (add_idx < 18816) {
        mobilenetv20_features_linearbottleneck11_elemwise_add0[add_idx] = mobilenetv20_features_linearbottleneck11_batchnorm2_fwd[add_idx] + mobilenetv20_features_linearbottleneck10_batchnorm2_fwd[add_idx];
        add_idx = add_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck12_conv0_fwd
    // Weight shape: [576, 96, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck12_conv0_fwd_size = 55296;
    var weights_mobilenetv20_features_linearbottleneck12_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck12_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 439040
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck12_conv0_fwd_size) {
        if (439040 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck12_conv0_fwd[wi] = weights[439040 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck12_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck12_conv0_fwd
    var mobilenetv20_features_linearbottleneck12_conv0_fwd_size = 112896;
    var mobilenetv20_features_linearbottleneck12_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck12_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 576 x 96
    //   B (im2col):  96 x 196 
    //   C (output):  576 x 196
    
    // Allocate im2col buffer
    var col_size = 18816;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck11_elemwise_add0, 1, 14, 14, 96,
                  1, 1, 1, 1, 0, 0, 
                  14, 14, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([112896]);
    
    // GEMM call: C = A * B
    // A = weights [576 x 96]
    // B = col_buffer [96 x 196]  
    // C = gemm_output [576 x 196]
    sb_gemm_f32(576, 196, 96,
                weights_mobilenetv20_features_linearbottleneck12_conv0_fwd, 96,
                col_buffer, 196, 
                gemm_output, 196);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 14) {
        var out_x = 0;
        while (out_x < 14) {
            var out_ch = 0;
            while (out_ch < 576) {
                var gemm_idx = out_ch * 196 + out_y * 14 + out_x;
                var out_offset = nhwc_offset(1, 14, 14, 576, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck12_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck12_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd = array<f32, simd=auto>([576]);
    var bn_bias_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd = array<f32, simd=auto>([576]);
    var bn_mean_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd = array<f32, simd=auto>([576]);
    var bn_var_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd = array<f32, simd=auto>([576]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 576) {
        if (494336 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd[bni] = weights[494336 + bni];
        }
        if (494912 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd[bni] = weights[494912 + bni];
        }
        if (495488 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd[bni] = weights[495488 + bni];
        }
        if (496064 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd[bni] = weights[496064 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck12_batchnorm0_fwd = mobilenetv20_features_linearbottleneck12_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 112896) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 576;
        var val = mobilenetv20_features_linearbottleneck12_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck12_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck12_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck12_relu0_fwd
    var mobilenetv20_features_linearbottleneck12_relu0_fwd = mobilenetv20_features_linearbottleneck12_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 112896) {
        var val = mobilenetv20_features_linearbottleneck12_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck12_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck12_conv1_fwd
    // Weight shape: [576, 1, 3, 3], Groups: 576
    var weights_mobilenetv20_features_linearbottleneck12_conv1_fwd_size = 5184;
    var weights_mobilenetv20_features_linearbottleneck12_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck12_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 496640
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck12_conv1_fwd_size) {
        if (496640 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck12_conv1_fwd[wi] = weights[496640 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck12_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck12_conv1_fwd
    var mobilenetv20_features_linearbottleneck12_conv1_fwd_size = 112896;
    var mobilenetv20_features_linearbottleneck12_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck12_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 576)
    var batch = 0;
    var out_y = 0;
    while (out_y < 14) {
        var out_x = 0;
        while (out_x < 14) {
            var out_ch = 0;
            while (out_ch < 576) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 14) {
                                if (in_x >= 0) {
                                    if (in_x < 14) {
                                        var in_offset = nhwc_offset(1, 14, 14, 576, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck12_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck12_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 14, 14, 576, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck12_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck12_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd = array<f32, simd=auto>([576]);
    var bn_bias_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd = array<f32, simd=auto>([576]);
    var bn_mean_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd = array<f32, simd=auto>([576]);
    var bn_var_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd = array<f32, simd=auto>([576]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 576) {
        if (501824 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd[bni] = weights[501824 + bni];
        }
        if (502400 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd[bni] = weights[502400 + bni];
        }
        if (502976 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd[bni] = weights[502976 + bni];
        }
        if (503552 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd[bni] = weights[503552 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck12_batchnorm1_fwd = mobilenetv20_features_linearbottleneck12_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 112896) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 576;
        var val = mobilenetv20_features_linearbottleneck12_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck12_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck12_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck12_relu1_fwd
    var mobilenetv20_features_linearbottleneck12_relu1_fwd = mobilenetv20_features_linearbottleneck12_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 112896) {
        var val = mobilenetv20_features_linearbottleneck12_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck12_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck12_conv2_fwd
    // Weight shape: [96, 576, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck12_conv2_fwd_size = 55296;
    var weights_mobilenetv20_features_linearbottleneck12_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck12_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 504128
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck12_conv2_fwd_size) {
        if (504128 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck12_conv2_fwd[wi] = weights[504128 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck12_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck12_conv2_fwd
    var mobilenetv20_features_linearbottleneck12_conv2_fwd_size = 18816;
    var mobilenetv20_features_linearbottleneck12_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck12_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 96 x 576
    //   B (im2col):  576 x 196 
    //   C (output):  96 x 196
    
    // Allocate im2col buffer
    var col_size = 112896;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck12_relu1_fwd, 1, 14, 14, 576,
                  1, 1, 1, 1, 0, 0, 
                  14, 14, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([18816]);
    
    // GEMM call: C = A * B
    // A = weights [96 x 576]
    // B = col_buffer [576 x 196]  
    // C = gemm_output [96 x 196]
    sb_gemm_f32(96, 196, 576,
                weights_mobilenetv20_features_linearbottleneck12_conv2_fwd, 576,
                col_buffer, 196, 
                gemm_output, 196);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 14) {
        var out_x = 0;
        while (out_x < 14) {
            var out_ch = 0;
            while (out_ch < 96) {
                var gemm_idx = out_ch * 196 + out_y * 14 + out_x;
                var out_offset = nhwc_offset(1, 14, 14, 96, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck12_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck12_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd = array<f32, simd=auto>([96]);
    var bn_bias_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd = array<f32, simd=auto>([96]);
    var bn_mean_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd = array<f32, simd=auto>([96]);
    var bn_var_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd = array<f32, simd=auto>([96]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 96) {
        if (559424 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd[bni] = weights[559424 + bni];
        }
        if (559520 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd[bni] = weights[559520 + bni];
        }
        if (559616 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd[bni] = weights[559616 + bni];
        }
        if (559712 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd[bni] = weights[559712 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck12_batchnorm2_fwd = mobilenetv20_features_linearbottleneck12_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 18816) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 96;
        var val = mobilenetv20_features_linearbottleneck12_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck12_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck12_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Add layer (residual): mobilenetv20_features_linearbottleneck12_elemwise_add0
    var mobilenetv20_features_linearbottleneck12_elemwise_add0_size = 18816;
    var mobilenetv20_features_linearbottleneck12_elemwise_add0 = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck12_elemwise_add0_size]);
    
    var add_idx = 0;
    while (add_idx < 18816) {
        mobilenetv20_features_linearbottleneck12_elemwise_add0[add_idx] = mobilenetv20_features_linearbottleneck12_batchnorm2_fwd[add_idx] + mobilenetv20_features_linearbottleneck11_elemwise_add0[add_idx];
        add_idx = add_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck13_conv0_fwd
    // Weight shape: [576, 96, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck13_conv0_fwd_size = 55296;
    var weights_mobilenetv20_features_linearbottleneck13_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck13_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 559808
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck13_conv0_fwd_size) {
        if (559808 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck13_conv0_fwd[wi] = weights[559808 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck13_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck13_conv0_fwd
    var mobilenetv20_features_linearbottleneck13_conv0_fwd_size = 112896;
    var mobilenetv20_features_linearbottleneck13_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck13_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 576 x 96
    //   B (im2col):  96 x 196 
    //   C (output):  576 x 196
    
    // Allocate im2col buffer
    var col_size = 18816;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck12_elemwise_add0, 1, 14, 14, 96,
                  1, 1, 1, 1, 0, 0, 
                  14, 14, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([112896]);
    
    // GEMM call: C = A * B
    // A = weights [576 x 96]
    // B = col_buffer [96 x 196]  
    // C = gemm_output [576 x 196]
    sb_gemm_f32(576, 196, 96,
                weights_mobilenetv20_features_linearbottleneck13_conv0_fwd, 96,
                col_buffer, 196, 
                gemm_output, 196);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 14) {
        var out_x = 0;
        while (out_x < 14) {
            var out_ch = 0;
            while (out_ch < 576) {
                var gemm_idx = out_ch * 196 + out_y * 14 + out_x;
                var out_offset = nhwc_offset(1, 14, 14, 576, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck13_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck13_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd = array<f32, simd=auto>([576]);
    var bn_bias_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd = array<f32, simd=auto>([576]);
    var bn_mean_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd = array<f32, simd=auto>([576]);
    var bn_var_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd = array<f32, simd=auto>([576]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 576) {
        if (615104 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd[bni] = weights[615104 + bni];
        }
        if (615680 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd[bni] = weights[615680 + bni];
        }
        if (616256 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd[bni] = weights[616256 + bni];
        }
        if (616832 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd[bni] = weights[616832 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck13_batchnorm0_fwd = mobilenetv20_features_linearbottleneck13_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 112896) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 576;
        var val = mobilenetv20_features_linearbottleneck13_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck13_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck13_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck13_relu0_fwd
    var mobilenetv20_features_linearbottleneck13_relu0_fwd = mobilenetv20_features_linearbottleneck13_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 112896) {
        var val = mobilenetv20_features_linearbottleneck13_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck13_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck13_conv1_fwd
    // Weight shape: [576, 1, 3, 3], Groups: 576
    var weights_mobilenetv20_features_linearbottleneck13_conv1_fwd_size = 5184;
    var weights_mobilenetv20_features_linearbottleneck13_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck13_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 617408
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck13_conv1_fwd_size) {
        if (617408 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck13_conv1_fwd[wi] = weights[617408 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck13_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck13_conv1_fwd
    var mobilenetv20_features_linearbottleneck13_conv1_fwd_size = 28224;
    var mobilenetv20_features_linearbottleneck13_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck13_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 576)
    var batch = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 576) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 2 + ky - 1;
                        var in_x = out_x * 2 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 14) {
                                if (in_x >= 0) {
                                    if (in_x < 14) {
                                        var in_offset = nhwc_offset(1, 14, 14, 576, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck13_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck13_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 7, 7, 576, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck13_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck13_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd = array<f32, simd=auto>([576]);
    var bn_bias_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd = array<f32, simd=auto>([576]);
    var bn_mean_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd = array<f32, simd=auto>([576]);
    var bn_var_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd = array<f32, simd=auto>([576]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 576) {
        if (622592 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd[bni] = weights[622592 + bni];
        }
        if (623168 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd[bni] = weights[623168 + bni];
        }
        if (623744 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd[bni] = weights[623744 + bni];
        }
        if (624320 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd[bni] = weights[624320 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck13_batchnorm1_fwd = mobilenetv20_features_linearbottleneck13_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 28224) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 576;
        var val = mobilenetv20_features_linearbottleneck13_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck13_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck13_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck13_relu1_fwd
    var mobilenetv20_features_linearbottleneck13_relu1_fwd = mobilenetv20_features_linearbottleneck13_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 28224) {
        var val = mobilenetv20_features_linearbottleneck13_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck13_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck13_conv2_fwd
    // Weight shape: [160, 576, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck13_conv2_fwd_size = 92160;
    var weights_mobilenetv20_features_linearbottleneck13_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck13_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 624896
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck13_conv2_fwd_size) {
        if (624896 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck13_conv2_fwd[wi] = weights[624896 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck13_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck13_conv2_fwd
    var mobilenetv20_features_linearbottleneck13_conv2_fwd_size = 7840;
    var mobilenetv20_features_linearbottleneck13_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck13_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 160 x 576
    //   B (im2col):  576 x 49 
    //   C (output):  160 x 49
    
    // Allocate im2col buffer
    var col_size = 28224;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck13_relu1_fwd, 1, 7, 7, 576,
                  1, 1, 1, 1, 0, 0, 
                  7, 7, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([7840]);
    
    // GEMM call: C = A * B
    // A = weights [160 x 576]
    // B = col_buffer [576 x 49]  
    // C = gemm_output [160 x 49]
    sb_gemm_f32(160, 49, 576,
                weights_mobilenetv20_features_linearbottleneck13_conv2_fwd, 576,
                col_buffer, 49, 
                gemm_output, 49);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 160) {
                var gemm_idx = out_ch * 49 + out_y * 7 + out_x;
                var out_offset = nhwc_offset(1, 7, 7, 160, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck13_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck13_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd = array<f32, simd=auto>([160]);
    var bn_bias_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd = array<f32, simd=auto>([160]);
    var bn_mean_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd = array<f32, simd=auto>([160]);
    var bn_var_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd = array<f32, simd=auto>([160]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 160) {
        if (717056 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd[bni] = weights[717056 + bni];
        }
        if (717216 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd[bni] = weights[717216 + bni];
        }
        if (717376 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd[bni] = weights[717376 + bni];
        }
        if (717536 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd[bni] = weights[717536 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck13_batchnorm2_fwd = mobilenetv20_features_linearbottleneck13_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 7840) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 160;
        var val = mobilenetv20_features_linearbottleneck13_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck13_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck13_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck14_conv0_fwd
    // Weight shape: [960, 160, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck14_conv0_fwd_size = 153600;
    var weights_mobilenetv20_features_linearbottleneck14_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck14_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 717696
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck14_conv0_fwd_size) {
        if (717696 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck14_conv0_fwd[wi] = weights[717696 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck14_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck14_conv0_fwd
    var mobilenetv20_features_linearbottleneck14_conv0_fwd_size = 47040;
    var mobilenetv20_features_linearbottleneck14_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck14_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 960 x 160
    //   B (im2col):  160 x 49 
    //   C (output):  960 x 49
    
    // Allocate im2col buffer
    var col_size = 7840;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck13_batchnorm2_fwd, 1, 7, 7, 160,
                  1, 1, 1, 1, 0, 0, 
                  7, 7, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([47040]);
    
    // GEMM call: C = A * B
    // A = weights [960 x 160]
    // B = col_buffer [160 x 49]  
    // C = gemm_output [960 x 49]
    sb_gemm_f32(960, 49, 160,
                weights_mobilenetv20_features_linearbottleneck14_conv0_fwd, 160,
                col_buffer, 49, 
                gemm_output, 49);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 960) {
                var gemm_idx = out_ch * 49 + out_y * 7 + out_x;
                var out_offset = nhwc_offset(1, 7, 7, 960, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck14_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck14_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd = array<f32, simd=auto>([960]);
    var bn_bias_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd = array<f32, simd=auto>([960]);
    var bn_mean_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd = array<f32, simd=auto>([960]);
    var bn_var_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd = array<f32, simd=auto>([960]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 960) {
        if (871296 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd[bni] = weights[871296 + bni];
        }
        if (872256 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd[bni] = weights[872256 + bni];
        }
        if (873216 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd[bni] = weights[873216 + bni];
        }
        if (874176 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd[bni] = weights[874176 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck14_batchnorm0_fwd = mobilenetv20_features_linearbottleneck14_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 47040) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 960;
        var val = mobilenetv20_features_linearbottleneck14_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck14_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck14_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck14_relu0_fwd
    var mobilenetv20_features_linearbottleneck14_relu0_fwd = mobilenetv20_features_linearbottleneck14_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 47040) {
        var val = mobilenetv20_features_linearbottleneck14_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck14_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck14_conv1_fwd
    // Weight shape: [960, 1, 3, 3], Groups: 960
    var weights_mobilenetv20_features_linearbottleneck14_conv1_fwd_size = 8640;
    var weights_mobilenetv20_features_linearbottleneck14_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck14_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 875136
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck14_conv1_fwd_size) {
        if (875136 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck14_conv1_fwd[wi] = weights[875136 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck14_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck14_conv1_fwd
    var mobilenetv20_features_linearbottleneck14_conv1_fwd_size = 47040;
    var mobilenetv20_features_linearbottleneck14_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck14_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 960)
    var batch = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 960) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 7) {
                                if (in_x >= 0) {
                                    if (in_x < 7) {
                                        var in_offset = nhwc_offset(1, 7, 7, 960, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck14_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck14_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 7, 7, 960, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck14_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck14_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd = array<f32, simd=auto>([960]);
    var bn_bias_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd = array<f32, simd=auto>([960]);
    var bn_mean_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd = array<f32, simd=auto>([960]);
    var bn_var_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd = array<f32, simd=auto>([960]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 960) {
        if (883776 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd[bni] = weights[883776 + bni];
        }
        if (884736 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd[bni] = weights[884736 + bni];
        }
        if (885696 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd[bni] = weights[885696 + bni];
        }
        if (886656 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd[bni] = weights[886656 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck14_batchnorm1_fwd = mobilenetv20_features_linearbottleneck14_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 47040) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 960;
        var val = mobilenetv20_features_linearbottleneck14_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck14_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck14_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck14_relu1_fwd
    var mobilenetv20_features_linearbottleneck14_relu1_fwd = mobilenetv20_features_linearbottleneck14_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 47040) {
        var val = mobilenetv20_features_linearbottleneck14_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck14_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck14_conv2_fwd
    // Weight shape: [160, 960, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck14_conv2_fwd_size = 153600;
    var weights_mobilenetv20_features_linearbottleneck14_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck14_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 887616
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck14_conv2_fwd_size) {
        if (887616 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck14_conv2_fwd[wi] = weights[887616 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck14_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck14_conv2_fwd
    var mobilenetv20_features_linearbottleneck14_conv2_fwd_size = 7840;
    var mobilenetv20_features_linearbottleneck14_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck14_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 160 x 960
    //   B (im2col):  960 x 49 
    //   C (output):  160 x 49
    
    // Allocate im2col buffer
    var col_size = 47040;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck14_relu1_fwd, 1, 7, 7, 960,
                  1, 1, 1, 1, 0, 0, 
                  7, 7, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([7840]);
    
    // GEMM call: C = A * B
    // A = weights [160 x 960]
    // B = col_buffer [960 x 49]  
    // C = gemm_output [160 x 49]
    sb_gemm_f32(160, 49, 960,
                weights_mobilenetv20_features_linearbottleneck14_conv2_fwd, 960,
                col_buffer, 49, 
                gemm_output, 49);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 160) {
                var gemm_idx = out_ch * 49 + out_y * 7 + out_x;
                var out_offset = nhwc_offset(1, 7, 7, 160, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck14_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck14_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd = array<f32, simd=auto>([160]);
    var bn_bias_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd = array<f32, simd=auto>([160]);
    var bn_mean_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd = array<f32, simd=auto>([160]);
    var bn_var_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd = array<f32, simd=auto>([160]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 160) {
        if (1041216 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd[bni] = weights[1041216 + bni];
        }
        if (1041376 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd[bni] = weights[1041376 + bni];
        }
        if (1041536 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd[bni] = weights[1041536 + bni];
        }
        if (1041696 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd[bni] = weights[1041696 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck14_batchnorm2_fwd = mobilenetv20_features_linearbottleneck14_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 7840) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 160;
        var val = mobilenetv20_features_linearbottleneck14_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck14_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck14_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Add layer (residual): mobilenetv20_features_linearbottleneck14_elemwise_add0
    var mobilenetv20_features_linearbottleneck14_elemwise_add0_size = 7840;
    var mobilenetv20_features_linearbottleneck14_elemwise_add0 = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck14_elemwise_add0_size]);
    
    var add_idx = 0;
    while (add_idx < 7840) {
        mobilenetv20_features_linearbottleneck14_elemwise_add0[add_idx] = mobilenetv20_features_linearbottleneck14_batchnorm2_fwd[add_idx] + mobilenetv20_features_linearbottleneck13_batchnorm2_fwd[add_idx];
        add_idx = add_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck15_conv0_fwd
    // Weight shape: [960, 160, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck15_conv0_fwd_size = 153600;
    var weights_mobilenetv20_features_linearbottleneck15_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck15_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 1041856
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck15_conv0_fwd_size) {
        if (1041856 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck15_conv0_fwd[wi] = weights[1041856 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck15_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck15_conv0_fwd
    var mobilenetv20_features_linearbottleneck15_conv0_fwd_size = 47040;
    var mobilenetv20_features_linearbottleneck15_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck15_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 960 x 160
    //   B (im2col):  160 x 49 
    //   C (output):  960 x 49
    
    // Allocate im2col buffer
    var col_size = 7840;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck14_elemwise_add0, 1, 7, 7, 160,
                  1, 1, 1, 1, 0, 0, 
                  7, 7, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([47040]);
    
    // GEMM call: C = A * B
    // A = weights [960 x 160]
    // B = col_buffer [160 x 49]  
    // C = gemm_output [960 x 49]
    sb_gemm_f32(960, 49, 160,
                weights_mobilenetv20_features_linearbottleneck15_conv0_fwd, 160,
                col_buffer, 49, 
                gemm_output, 49);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 960) {
                var gemm_idx = out_ch * 49 + out_y * 7 + out_x;
                var out_offset = nhwc_offset(1, 7, 7, 960, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck15_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck15_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd = array<f32, simd=auto>([960]);
    var bn_bias_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd = array<f32, simd=auto>([960]);
    var bn_mean_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd = array<f32, simd=auto>([960]);
    var bn_var_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd = array<f32, simd=auto>([960]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 960) {
        if (1195456 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd[bni] = weights[1195456 + bni];
        }
        if (1196416 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd[bni] = weights[1196416 + bni];
        }
        if (1197376 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd[bni] = weights[1197376 + bni];
        }
        if (1198336 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd[bni] = weights[1198336 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck15_batchnorm0_fwd = mobilenetv20_features_linearbottleneck15_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 47040) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 960;
        var val = mobilenetv20_features_linearbottleneck15_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck15_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck15_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck15_relu0_fwd
    var mobilenetv20_features_linearbottleneck15_relu0_fwd = mobilenetv20_features_linearbottleneck15_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 47040) {
        var val = mobilenetv20_features_linearbottleneck15_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck15_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck15_conv1_fwd
    // Weight shape: [960, 1, 3, 3], Groups: 960
    var weights_mobilenetv20_features_linearbottleneck15_conv1_fwd_size = 8640;
    var weights_mobilenetv20_features_linearbottleneck15_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck15_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 1199296
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck15_conv1_fwd_size) {
        if (1199296 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck15_conv1_fwd[wi] = weights[1199296 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck15_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck15_conv1_fwd
    var mobilenetv20_features_linearbottleneck15_conv1_fwd_size = 47040;
    var mobilenetv20_features_linearbottleneck15_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck15_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 960)
    var batch = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 960) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 7) {
                                if (in_x >= 0) {
                                    if (in_x < 7) {
                                        var in_offset = nhwc_offset(1, 7, 7, 960, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck15_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck15_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 7, 7, 960, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck15_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck15_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd = array<f32, simd=auto>([960]);
    var bn_bias_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd = array<f32, simd=auto>([960]);
    var bn_mean_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd = array<f32, simd=auto>([960]);
    var bn_var_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd = array<f32, simd=auto>([960]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 960) {
        if (1207936 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd[bni] = weights[1207936 + bni];
        }
        if (1208896 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd[bni] = weights[1208896 + bni];
        }
        if (1209856 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd[bni] = weights[1209856 + bni];
        }
        if (1210816 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd[bni] = weights[1210816 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck15_batchnorm1_fwd = mobilenetv20_features_linearbottleneck15_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 47040) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 960;
        var val = mobilenetv20_features_linearbottleneck15_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck15_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck15_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck15_relu1_fwd
    var mobilenetv20_features_linearbottleneck15_relu1_fwd = mobilenetv20_features_linearbottleneck15_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 47040) {
        var val = mobilenetv20_features_linearbottleneck15_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck15_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck15_conv2_fwd
    // Weight shape: [160, 960, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck15_conv2_fwd_size = 153600;
    var weights_mobilenetv20_features_linearbottleneck15_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck15_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 1211776
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck15_conv2_fwd_size) {
        if (1211776 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck15_conv2_fwd[wi] = weights[1211776 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck15_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck15_conv2_fwd
    var mobilenetv20_features_linearbottleneck15_conv2_fwd_size = 7840;
    var mobilenetv20_features_linearbottleneck15_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck15_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 160 x 960
    //   B (im2col):  960 x 49 
    //   C (output):  160 x 49
    
    // Allocate im2col buffer
    var col_size = 47040;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck15_relu1_fwd, 1, 7, 7, 960,
                  1, 1, 1, 1, 0, 0, 
                  7, 7, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([7840]);
    
    // GEMM call: C = A * B
    // A = weights [160 x 960]
    // B = col_buffer [960 x 49]  
    // C = gemm_output [160 x 49]
    sb_gemm_f32(160, 49, 960,
                weights_mobilenetv20_features_linearbottleneck15_conv2_fwd, 960,
                col_buffer, 49, 
                gemm_output, 49);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 160) {
                var gemm_idx = out_ch * 49 + out_y * 7 + out_x;
                var out_offset = nhwc_offset(1, 7, 7, 160, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck15_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck15_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd = array<f32, simd=auto>([160]);
    var bn_bias_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd = array<f32, simd=auto>([160]);
    var bn_mean_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd = array<f32, simd=auto>([160]);
    var bn_var_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd = array<f32, simd=auto>([160]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 160) {
        if (1365376 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd[bni] = weights[1365376 + bni];
        }
        if (1365536 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd[bni] = weights[1365536 + bni];
        }
        if (1365696 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd[bni] = weights[1365696 + bni];
        }
        if (1365856 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd[bni] = weights[1365856 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck15_batchnorm2_fwd = mobilenetv20_features_linearbottleneck15_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 7840) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 160;
        var val = mobilenetv20_features_linearbottleneck15_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck15_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck15_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Add layer (residual): mobilenetv20_features_linearbottleneck15_elemwise_add0
    var mobilenetv20_features_linearbottleneck15_elemwise_add0_size = 7840;
    var mobilenetv20_features_linearbottleneck15_elemwise_add0 = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck15_elemwise_add0_size]);
    
    var add_idx = 0;
    while (add_idx < 7840) {
        mobilenetv20_features_linearbottleneck15_elemwise_add0[add_idx] = mobilenetv20_features_linearbottleneck15_batchnorm2_fwd[add_idx] + mobilenetv20_features_linearbottleneck14_elemwise_add0[add_idx];
        add_idx = add_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck16_conv0_fwd
    // Weight shape: [960, 160, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck16_conv0_fwd_size = 153600;
    var weights_mobilenetv20_features_linearbottleneck16_conv0_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck16_conv0_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 1366016
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck16_conv0_fwd_size) {
        if (1366016 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck16_conv0_fwd[wi] = weights[1366016 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck16_conv0_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck16_conv0_fwd
    var mobilenetv20_features_linearbottleneck16_conv0_fwd_size = 47040;
    var mobilenetv20_features_linearbottleneck16_conv0_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck16_conv0_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 960 x 160
    //   B (im2col):  160 x 49 
    //   C (output):  960 x 49
    
    // Allocate im2col buffer
    var col_size = 7840;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck15_elemwise_add0, 1, 7, 7, 160,
                  1, 1, 1, 1, 0, 0, 
                  7, 7, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([47040]);
    
    // GEMM call: C = A * B
    // A = weights [960 x 160]
    // B = col_buffer [160 x 49]  
    // C = gemm_output [960 x 49]
    sb_gemm_f32(960, 49, 160,
                weights_mobilenetv20_features_linearbottleneck16_conv0_fwd, 160,
                col_buffer, 49, 
                gemm_output, 49);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 960) {
                var gemm_idx = out_ch * 49 + out_y * 7 + out_x;
                var out_offset = nhwc_offset(1, 7, 7, 960, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck16_conv0_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck16_batchnorm0_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd = array<f32, simd=auto>([960]);
    var bn_bias_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd = array<f32, simd=auto>([960]);
    var bn_mean_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd = array<f32, simd=auto>([960]);
    var bn_var_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd = array<f32, simd=auto>([960]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 960) {
        if (1519616 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd[bni] = weights[1519616 + bni];
        }
        if (1520576 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd[bni] = weights[1520576 + bni];
        }
        if (1521536 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd[bni] = weights[1521536 + bni];
        }
        if (1522496 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd[bni] = weights[1522496 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck16_batchnorm0_fwd = mobilenetv20_features_linearbottleneck16_conv0_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 47040) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 960;
        var val = mobilenetv20_features_linearbottleneck16_conv0_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck16_batchnorm0_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck16_batchnorm0_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck16_relu0_fwd
    var mobilenetv20_features_linearbottleneck16_relu0_fwd = mobilenetv20_features_linearbottleneck16_batchnorm0_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 47040) {
        var val = mobilenetv20_features_linearbottleneck16_batchnorm0_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck16_relu0_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck16_conv1_fwd
    // Weight shape: [960, 1, 3, 3], Groups: 960
    var weights_mobilenetv20_features_linearbottleneck16_conv1_fwd_size = 8640;
    var weights_mobilenetv20_features_linearbottleneck16_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck16_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 1523456
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck16_conv1_fwd_size) {
        if (1523456 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck16_conv1_fwd[wi] = weights[1523456 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck16_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck16_conv1_fwd
    var mobilenetv20_features_linearbottleneck16_conv1_fwd_size = 47040;
    var mobilenetv20_features_linearbottleneck16_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck16_conv1_fwd_size]);
    
    // Convolution computation
    // Depthwise convolution (groups = 960)
    var batch = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 960) {
                var sum = 0.0;
                var in_ch = out_ch;  // For depthwise, input channel = output channel
                
                var ky = 0;
                while (ky < 3) {
                    var kx = 0;
                    while (kx < 3) {
                        var in_y = out_y * 1 + ky - 1;
                        var in_x = out_x * 1 + kx - 1;
                        
                        // Bounds check
                        if (in_y >= 0) {
                            if (in_y < 7) {
                                if (in_x >= 0) {
                                    if (in_x < 7) {
                                        var in_offset = nhwc_offset(1, 7, 7, 960, batch, in_y, in_x, in_ch);
                                        var weight_idx = out_ch * 9 + ky * 3 + kx;
                                        sum = sum + mobilenetv20_features_linearbottleneck16_relu0_fwd[in_offset] * weights_mobilenetv20_features_linearbottleneck16_conv1_fwd[weight_idx];
                                    }
                                }
                            }
                        }
                        kx = kx + 1;
                    }
                    ky = ky + 1;
                }
                
                var out_offset = nhwc_offset(1, 7, 7, 960, batch, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck16_conv1_fwd[out_offset] = sum;
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck16_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd = array<f32, simd=auto>([960]);
    var bn_bias_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd = array<f32, simd=auto>([960]);
    var bn_mean_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd = array<f32, simd=auto>([960]);
    var bn_var_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd = array<f32, simd=auto>([960]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 960) {
        if (1532096 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd[bni] = weights[1532096 + bni];
        }
        if (1533056 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd[bni] = weights[1533056 + bni];
        }
        if (1534016 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd[bni] = weights[1534016 + bni];
        }
        if (1534976 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd[bni] = weights[1534976 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck16_batchnorm1_fwd = mobilenetv20_features_linearbottleneck16_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 47040) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 960;
        var val = mobilenetv20_features_linearbottleneck16_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck16_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck16_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_linearbottleneck16_relu1_fwd
    var mobilenetv20_features_linearbottleneck16_relu1_fwd = mobilenetv20_features_linearbottleneck16_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 47040) {
        var val = mobilenetv20_features_linearbottleneck16_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_linearbottleneck16_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // Conv layer: mobilenetv20_features_linearbottleneck16_conv2_fwd
    // Weight shape: [320, 960, 1, 1], Groups: 1
    var weights_mobilenetv20_features_linearbottleneck16_conv2_fwd_size = 307200;
    var weights_mobilenetv20_features_linearbottleneck16_conv2_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_linearbottleneck16_conv2_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 1535936
    var wi = 0;
    while (wi < weights_mobilenetv20_features_linearbottleneck16_conv2_fwd_size) {
        if (1535936 + wi < weight_count) {
            weights_mobilenetv20_features_linearbottleneck16_conv2_fwd[wi] = weights[1535936 + wi];
        } else {
            weights_mobilenetv20_features_linearbottleneck16_conv2_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_linearbottleneck16_conv2_fwd
    var mobilenetv20_features_linearbottleneck16_conv2_fwd_size = 15680;
    var mobilenetv20_features_linearbottleneck16_conv2_fwd = array<f32, simd=auto>([mobilenetv20_features_linearbottleneck16_conv2_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 320 x 960
    //   B (im2col):  960 x 49 
    //   C (output):  320 x 49
    
    // Allocate im2col buffer
    var col_size = 47040;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck16_relu1_fwd, 1, 7, 7, 960,
                  1, 1, 1, 1, 0, 0, 
                  7, 7, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([15680]);
    
    // GEMM call: C = A * B
    // A = weights [320 x 960]
    // B = col_buffer [960 x 49]  
    // C = gemm_output [320 x 49]
    sb_gemm_f32(320, 49, 960,
                weights_mobilenetv20_features_linearbottleneck16_conv2_fwd, 960,
                col_buffer, 49, 
                gemm_output, 49);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 320) {
                var gemm_idx = out_ch * 49 + out_y * 7 + out_x;
                var out_offset = nhwc_offset(1, 7, 7, 320, 0, out_y, out_x, out_ch);
                mobilenetv20_features_linearbottleneck16_conv2_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_linearbottleneck16_batchnorm2_fwd
    var bn_scale_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd = array<f32, simd=auto>([320]);
    var bn_bias_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd = array<f32, simd=auto>([320]);
    var bn_mean_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd = array<f32, simd=auto>([320]);
    var bn_var_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd = array<f32, simd=auto>([320]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 320) {
        if (1843136 + bni < weight_count) {
            bn_scale_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd[bni] = weights[1843136 + bni];
        }
        if (1843456 + bni < weight_count) {
            bn_bias_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd[bni] = weights[1843456 + bni];
        }
        if (1843776 + bni < weight_count) {
            bn_mean_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd[bni] = weights[1843776 + bni];
        }
        if (1844096 + bni < weight_count) {
            bn_var_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd[bni] = weights[1844096 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_linearbottleneck16_batchnorm2_fwd = mobilenetv20_features_linearbottleneck16_conv2_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 15680) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 320;
        var val = mobilenetv20_features_linearbottleneck16_conv2_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_linearbottleneck16_batchnorm2_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_linearbottleneck16_batchnorm2_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // Conv layer: mobilenetv20_features_conv1_fwd
    // Weight shape: [1280, 320, 1, 1], Groups: 1
    var weights_mobilenetv20_features_conv1_fwd_size = 409600;
    var weights_mobilenetv20_features_conv1_fwd = array<f32, simd=auto>([weights_mobilenetv20_features_conv1_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 1844416
    var wi = 0;
    while (wi < weights_mobilenetv20_features_conv1_fwd_size) {
        if (1844416 + wi < weight_count) {
            weights_mobilenetv20_features_conv1_fwd[wi] = weights[1844416 + wi];
        } else {
            weights_mobilenetv20_features_conv1_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_features_conv1_fwd
    var mobilenetv20_features_conv1_fwd_size = 62720;
    var mobilenetv20_features_conv1_fwd = array<f32, simd=auto>([mobilenetv20_features_conv1_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 1280 x 320
    //   B (im2col):  320 x 49 
    //   C (output):  1280 x 49
    
    // Allocate im2col buffer
    var col_size = 15680;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_linearbottleneck16_batchnorm2_fwd, 1, 7, 7, 320,
                  1, 1, 1, 1, 0, 0, 
                  7, 7, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([62720]);
    
    // GEMM call: C = A * B
    // A = weights [1280 x 320]
    // B = col_buffer [320 x 49]  
    // C = gemm_output [1280 x 49]
    sb_gemm_f32(1280, 49, 320,
                weights_mobilenetv20_features_conv1_fwd, 320,
                col_buffer, 49, 
                gemm_output, 49);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 7) {
        var out_x = 0;
        while (out_x < 7) {
            var out_ch = 0;
            while (out_ch < 1280) {
                var gemm_idx = out_ch * 49 + out_y * 7 + out_x;
                var out_offset = nhwc_offset(1, 7, 7, 1280, 0, out_y, out_x, out_ch);
                mobilenetv20_features_conv1_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // BatchNorm layer: mobilenetv20_features_batchnorm1_fwd
    var bn_scale_mobilenetv20_features_batchnorm1_fwd = array<f32, simd=auto>([1280]);
    var bn_bias_mobilenetv20_features_batchnorm1_fwd = array<f32, simd=auto>([1280]);
    var bn_mean_mobilenetv20_features_batchnorm1_fwd = array<f32, simd=auto>([1280]);
    var bn_var_mobilenetv20_features_batchnorm1_fwd = array<f32, simd=auto>([1280]);
    
    // Initialize batch norm parameters
    // Load batch norm parameters from passed weights
    var bni = 0;
    while (bni < 1280) {
        if (2254016 + bni < weight_count) {
            bn_scale_mobilenetv20_features_batchnorm1_fwd[bni] = weights[2254016 + bni];
        }
        if (2255296 + bni < weight_count) {
            bn_bias_mobilenetv20_features_batchnorm1_fwd[bni] = weights[2255296 + bni];
        }
        if (2256576 + bni < weight_count) {
            bn_mean_mobilenetv20_features_batchnorm1_fwd[bni] = weights[2256576 + bni];
        }
        if (2257856 + bni < weight_count) {
            bn_var_mobilenetv20_features_batchnorm1_fwd[bni] = weights[2257856 + bni];
        }
        bni = bni + 1;
    }
    
    // Create output tensor (reuse input for in-place operation)
    var mobilenetv20_features_batchnorm1_fwd = mobilenetv20_features_conv1_fwd;
    
    // Apply batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    var bn_idx = 0;
    while (bn_idx < 62720) {
        // Get channel index (last dimension in NHWC)
        var ch_idx = bn_idx % 1280;
        var val = mobilenetv20_features_conv1_fwd[bn_idx];
        var eps = 0.00001;
        
        var var_eps = bn_var_mobilenetv20_features_batchnorm1_fwd[ch_idx] + eps;
        if (var_eps < 0.000001) {
            var_eps = 0.000001;  // Prevent division by zero
        }
        
        var norm_val = (val - bn_mean_mobilenetv20_features_batchnorm1_fwd[ch_idx]) / var_eps * 
                       bn_scale_mobilenetv20_features_batchnorm1_fwd[ch_idx] + 
                       bn_bias_mobilenetv20_features_batchnorm1_fwd[ch_idx];
        
        // Clamp output to prevent overflow
        if (norm_val > 100.0) {
            norm_val = 100.0;
        }
        if (norm_val < -100.0) {
            norm_val = -100.0;
        }
        
        mobilenetv20_features_batchnorm1_fwd[bn_idx] = norm_val;
        bn_idx = bn_idx + 1;
    }
    // ReLU layer: mobilenetv20_features_relu1_fwd
    var mobilenetv20_features_relu1_fwd = mobilenetv20_features_batchnorm1_fwd;
    
    var relu_idx = 0;
    while (relu_idx < 62720) {
        var val = mobilenetv20_features_batchnorm1_fwd[relu_idx];
        if (val < 0.0) {
            val = 0.0;
        }
        mobilenetv20_features_relu1_fwd[relu_idx] = val;
        relu_idx = relu_idx + 1;
    }
    // GlobalAveragePool layer: mobilenetv20_features_pool0_fwd
    var mobilenetv20_features_pool0_fwd_size = 1280;
    var mobilenetv20_features_pool0_fwd = array<f32, simd=auto>([mobilenetv20_features_pool0_fwd_size]);
    
    var ch = 0;
    while (ch < 1280) {
        var sum = 0.0;
        var spatial_idx = 0;
        while (spatial_idx < 49) {
            var y = spatial_idx / 7;
            var x = spatial_idx % 7;
            var offset = nhwc_offset(1, 7, 7, 1280, 0, y, x, ch);
            sum = sum + mobilenetv20_features_relu1_fwd[offset];
            spatial_idx = spatial_idx + 1;
        }
        mobilenetv20_features_pool0_fwd[ch] = sum / 49.0;
        ch = ch + 1;
    }
    // Conv layer: mobilenetv20_output_pred_fwd
    // Weight shape: [1000, 1280, 1, 1], Groups: 1
    var weights_mobilenetv20_output_pred_fwd_size = 1280000;
    var weights_mobilenetv20_output_pred_fwd = array<f32, simd=auto>([weights_mobilenetv20_output_pred_fwd_size]);
    
    // Initialize weights (showing first few values)
    // Load weights from passed weights parameter at offset 2259136
    var wi = 0;
    while (wi < weights_mobilenetv20_output_pred_fwd_size) {
        if (2259136 + wi < weight_count) {
            weights_mobilenetv20_output_pred_fwd[wi] = weights[2259136 + wi];
        } else {
            weights_mobilenetv20_output_pred_fwd[wi] = 0.0;  // Safety fallback
        }
        wi = wi + 1;
    }
    
    // Output tensor: mobilenetv20_output_pred_fwd
    var mobilenetv20_output_pred_fwd_size = 1000;
    var mobilenetv20_output_pred_fwd = array<f32, simd=auto>([mobilenetv20_output_pred_fwd_size]);
    
    // Convolution computation
    // GEMM-based convolution using im2col transformation
    // Matrix dimensions: 
    //   A (weights): 1000 x 1280
    //   B (im2col):  1280 x 1 
    //   C (output):  1000 x 1
    
    // Allocate im2col buffer
    var col_size = 1280;
    var col_buffer = array<f32, simd=auto>([col_size]);
    
    // Transform input using im2col
    im2col_conv2d(mobilenetv20_features_pool0_fwd, 1, 1, 1, 1280,
                  1, 1, 1, 1, 0, 0, 
                  1, 1, col_buffer);
    
    // Reshape output for GEMM: [out_channels, out_h * out_w]
    var gemm_output = array<f32, simd=auto>([1000]);
    
    // GEMM call: C = A * B
    // A = weights [1000 x 1280]
    // B = col_buffer [1280 x 1]  
    // C = gemm_output [1000 x 1]
    sb_gemm_f32(1000, 1, 1280,
                weights_mobilenetv20_output_pred_fwd, 1280,
                col_buffer, 1, 
                gemm_output, 1);
    
    // Copy GEMM result to NHWC output format
    var reshape_idx = 0;
    var out_y = 0;
    while (out_y < 1) {
        var out_x = 0;
        while (out_x < 1) {
            var out_ch = 0;
            while (out_ch < 1000) {
                var gemm_idx = out_ch * 1 + out_y * 1 + out_x;
                var out_offset = nhwc_offset(1, 1, 1, 1000, 0, out_y, out_x, out_ch);
                mobilenetv20_output_pred_fwd[out_offset] = gemm_output[gemm_idx];
                out_ch = out_ch + 1;
            }
            out_x = out_x + 1;
        }
        out_y = out_y + 1;
    }
    // Reshape layer: mobilenetv20_output_flatten0_reshape0 (view change only)
    var mobilenetv20_output_flatten0_reshape0 = mobilenetv20_output_pred_fwd;

    // Find argmax (predicted class)
    var max_score = mobilenetv20_output_flatten0_reshape0[0];
    var max_class = 0;
    var i = 1;
    while (i < 1000) {
        if (mobilenetv20_output_flatten0_reshape0[i] > max_score) {
            max_score = mobilenetv20_output_flatten0_reshape0[i];
            max_class = i;
        }
        i = i + 1;
    }

    // Return the predicted class
    return max_class;
}

// NOTE: This function previously returned logits array but that doesn't work with -> var
// Now it returns the predicted class like mobilenet_inference_with_weights
fn get_logits_with_weights(f32[] weights, i32 weight_count, f32[] input_data, i32 input_size) -> f32 {
    return mobilenet_inference_core(weights, weight_count, input_data, input_size);
}

// Host-compatible function that returns argmax class as float
fn mobilenet_inference_with_weights(f32[] weights, i32 weight_count, f32[] input_data, i32 input_size) -> f32 {
    // mobilenet_inference_core now directly returns the predicted class
    return mobilenet_inference_core(weights, weight_count, input_data, input_size);
}


// Standalone version - ERROR: No weights provided!
fn mobilenet_inference() -> f32 {
    // This function should NOT be called - weights must be provided!
    return -999.0; // Error indicator
}

fn kernel_main() -> f32 {
    // This should NOT be used - only for backwards compatibility
    return mobilenet_inference();
}

// Main entry point that takes real weights and input from host  
fn kernel_main_with_weights(f32[] weights, i32 weight_count, f32[] input_data, i32 input_size) -> f32 {
    return mobilenet_inference_with_weights(weights, weight_count, input_data, input_size);
}

