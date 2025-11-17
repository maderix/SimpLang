// NHWC stride calculation helper
fn nhwc_offset(i32 n, i32 h, i32 w, i32 c,
               i32 batch, i32 height, i32 width, i32 channel) -> i32 {
    // NHWC layout: stride = [H*W*C, W*C, C, 1]
    var offset = batch * (h * w * c) +
                height * (w * c) +
                width * c +
                channel;
    return offset;
}

fn kernel_main() -> f32 {
    // Large tensor dimensions for stress testing
    var batch_size = 4;
    var height = 64;
    var width = 64; 
    var channels = 128;
    var total_elements = batch_size * height * width * channels;
    
    // Create large SIMD-optimized tensor
    var tensor_data = array<f32, simd=avx512>([total_elements]);
    
    // Test 1: Tensor initialization (write stress test)
    var init_start = 0;
    var b = 0;
    var h = 0;
    var w = 0;
    var c = 0;
    
    while (b < batch_size) {
        h = 0;
        while (h < height) {
            w = 0;
            while (w < width) {
                c = 0;
                while (c < channels) {
                    var offset = nhwc_offset(batch_size, height, width, channels, b, h, w, c);
                    var value = b * 1000.0 + h * 100.0 + w * 10.0 + c * 1.0;
                    tensor_data[offset] = value;
                    c = c + 1;
                }
                w = w + 1;
            }
            h = h + 1;
        }
        b = b + 1;
    }
    
    // Test 2: Tensor element-wise operations (computation stress test)
    var compute_sum = 0.0;
    var compute_count = 0;
    b = 0;
    while (b < batch_size) {
        h = 0;
        while (h < height) {
            w = 0;
            while (w < width) {
                c = 0;
                while (c < channels) {
                    var offset = nhwc_offset(batch_size, height, width, channels, b, h, w, c);
                    var value = tensor_data[offset];
                    
                    // Perform complex computation to stress test
                    var computed = value * value * 0.001 + value * 0.1 + 1.0;
                    computed = computed * computed * 0.5;
                    
                    compute_sum = compute_sum + computed;
                    compute_count = compute_count + 1;
                    c = c + 1;
                }
                w = w + 1;
            }
            h = h + 1;
        }
        b = b + 1;
    }
    
    // Test 3: Strided access patterns (memory access stress test)
    var stride_sum = 0.0;
    var stride_ops = 0;
    
    // Channel-wise stride test (access all elements of channel 0 across spatial dims)
    b = 0;
    while (b < batch_size) {
        h = 0;
        while (h < height) {
            w = 0;
            while (w < width) {
                var offset = nhwc_offset(batch_size, height, width, channels, b, h, w, 0);
                var value = tensor_data[offset];
                stride_sum = stride_sum + value;
                stride_ops = stride_ops + 1;
                w = w + 1;
            }
            h = h + 1;
        }
        b = b + 1;
    }
    
    // Test 4: Spatial reduction operations (reduction stress test)
    var reduce_sum = 0.0;
    var reduce_count = 0;
    
    // Reduce over height and width dimensions for each batch and channel
    b = 0;
    while (b < batch_size) {
        c = 0;
        while (c < channels) {
            var spatial_sum = 0.0;
            h = 0;
            while (h < height) {
                w = 0;
                while (w < width) {
                    var offset = nhwc_offset(batch_size, height, width, channels, b, h, w, c);
                    spatial_sum = spatial_sum + tensor_data[offset];
                    w = w + 1;
                }
                h = h + 1;
            }
            // Apply reduction operation
            var reduced_val = spatial_sum / (height * width);
            reduce_sum = reduce_sum + reduced_val * reduced_val;
            reduce_count = reduce_count + 1;
            c = c + 1;
        }
        b = b + 1;
    }
    
    // Final verification calculation
    var avg_compute = compute_sum / compute_count;
    var avg_stride = stride_sum / stride_ops;
    var avg_reduce = reduce_sum / reduce_count;
    
    // Return a composite result that exercises all operations
    var final_result = avg_compute * 0.001 + avg_stride * 0.0001 + avg_reduce * 0.00001;
    
    return final_result;
}