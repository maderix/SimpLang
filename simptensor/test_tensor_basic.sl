include "tensor_core.sl";

fn kernel_main() -> f32 {
    // Test basic tensor creation
    var result1 = tensor_f32(1, 8, 8, 3); // 1x8x8x3 tensor (192 elements)
    
    // Test NHWC offset calculation
    var offset1 = nhwc_offset(1, 8, 8, 3, 0, 0, 0, 0); // Should be 0
    var offset2 = nhwc_offset(1, 8, 8, 3, 0, 0, 0, 1); // Should be 1
    var offset3 = nhwc_offset(1, 8, 8, 3, 0, 0, 1, 0); // Should be 3
    var offset4 = nhwc_offset(1, 8, 8, 3, 0, 1, 0, 0); // Should be 24 (8*3)
    
    // Convert offsets to float for return
    var f_offset1 = offset1; // Should auto-convert i32 to f32
    var f_offset2 = offset2;
    var f_offset3 = offset3;
    var f_offset4 = offset4;
    
    // Return sum to verify calculations: 0 + 1 + 3 + 24 = 28
    return f_offset1 + f_offset2 + f_offset3 + f_offset4;
}