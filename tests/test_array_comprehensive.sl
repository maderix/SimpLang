fn kernel_main() {
    // Test 1: Float types
    var arr_f32 = array<f32>([3]);
    arr_f32[0] = 10.5;
    arr_f32[1] = 20.75;
    var f32_sum = arr_f32[0] + arr_f32[1];  // 31.25
    
    var arr_f64 = array<f64>([2]);
    arr_f64[0] = 100.125;
    arr_f64[1] = 200.875;
    var f64_result = arr_f64[1] - arr_f64[0];  // 100.75
    
    // Test 2: Signed integer types
    var arr_i8 = array<i8>([2]);
    arr_i8[0] = 127;
    arr_i8[1] = -128;
    var i8_elem = arr_i8[0];  // 127
    
    var arr_i16 = array<i16>([2]);
    arr_i16[0] = 32767;
    arr_i16[1] = -32768;
    var i16_elem = arr_i16[0];  // 32767
    
    var arr_i32 = array<i32>([3]);
    arr_i32[0] = 1000000;
    arr_i32[1] = -1000000;
    arr_i32[2] = arr_i32[0] + arr_i32[1];  // 0
    var i32_result = arr_i32[2];
    
    var arr_i64 = array<i64>([2]);
    arr_i64[0] = 9223372036854775807;
    arr_i64[1] = -9223372036854775808;
    var i64_elem = arr_i64[0];
    
    // Test 3: Unsigned integer types
    var arr_u8 = array<u8>([2]);
    arr_u8[0] = 255;
    arr_u8[1] = 0;
    var u8_elem = arr_u8[0];  // 255
    
    var arr_u16 = array<u16>([2]);
    arr_u16[0] = 65535;
    arr_u16[1] = 1000;
    var u16_elem = arr_u16[1];  // 1000
    
    var arr_u32 = array<u32>([2]);
    arr_u32[0] = 4294967295;
    arr_u32[1] = 12345;
    var u32_elem = arr_u32[1];  // 12345
    
    var arr_u64 = array<u64>([2]);
    arr_u64[0] = 18446744073709551615;
    arr_u64[1] = 99999;
    var u64_elem = arr_u64[1];  // 99999
    
    // Test 5: Dynamic type arrays (default to f64 for now)
    var arr_dynamic1 = array<f64>([4]);  // Using f64 as default dynamic type
    arr_dynamic1[0] = 42.0;  // Float assignment
    arr_dynamic1[1] = 84.0;
    var dyn_elem1 = arr_dynamic1[1];  // 84.0
    
    var arr_dynamic2 = array<i64>([3]);  // Using i64 for integer values
    arr_dynamic2[0] = 100;  // Integer assignment
    arr_dynamic2[1] = 200;
    var dyn_elem2 = arr_dynamic2[0];  // 100
    
    // Test 6: Mixed operations and type conversions
    var mixed_result = f32_sum + i32_result + u16_elem;  // 31.25 + 0 + 1000 = 1031.25
    
    // Test 7: Array indexing with variables
    var index = 1;
    var indexed_elem = arr_f32[index];  // Should get arr_f32[1] = 20.75
    
    // Test 8: Comprehensive indexing patterns
    var idx_test = array<f32>([10]);
    
    // Forward indexing
    idx_test[0] = 1.0;
    idx_test[1] = 2.0;
    idx_test[2] = 3.0;
    idx_test[3] = 4.0;
    idx_test[4] = 5.0;
    
    // Reverse indexing (from end)
    idx_test[9] = 10.0;
    idx_test[8] = 9.0;
    idx_test[7] = 8.0;
    idx_test[6] = 7.0;
    idx_test[5] = 6.0;
    
    // Random access pattern
    var random_sum = idx_test[3] + idx_test[7] + idx_test[1] + idx_test[9];
    // 4.0 + 8.0 + 2.0 + 10.0 = 24.0
    
    // Variable-based indexing with calculations
    var base_idx = 2;
    var offset = 3;
    var calc_idx = base_idx + offset;  // 5
    var calc_elem = idx_test[calc_idx];  // Should be 6.0
    
    // Test 9: Sequential access patterns
    var seq_array = array<i32>([5]);
    seq_array[0] = 10;
    seq_array[1] = seq_array[0] + 5;     // 15
    seq_array[2] = seq_array[1] + 5;     // 20
    seq_array[3] = seq_array[2] + 5;     // 25
    seq_array[4] = seq_array[3] + 5;     // 30
    
    // Access in different order
    var seq_sum = seq_array[4] + seq_array[2] + seq_array[0];
    // 30 + 20 + 10 = 60
    
    // Test 10: Stride patterns
    var stride_array = array<f32>([8]);
    // Fill every other element (stride 2)
    stride_array[0] = 100.0;
    stride_array[2] = 200.0;
    stride_array[4] = 300.0;
    stride_array[6] = 400.0;
    
    // Fill gaps (offset stride)
    stride_array[1] = 150.0;
    stride_array[3] = 250.0;
    stride_array[5] = 350.0;
    stride_array[7] = 450.0;
    
    // Access with stride pattern
    var stride_sum = stride_array[0] + stride_array[2] + stride_array[4] + stride_array[6];
    // 100 + 200 + 300 + 400 = 1000
    
    var offset_sum = stride_array[1] + stride_array[3] + stride_array[5] + stride_array[7];
    // 150 + 250 + 350 + 450 = 1200
    
    // Test 11: Boundary access
    var boundary_array = array<f32>([3]);
    boundary_array[0] = 111.0;  // First element
    boundary_array[2] = 333.0;  // Last element
    boundary_array[1] = boundary_array[0] + boundary_array[2];  // 444.0
    
    var boundary_result = boundary_array[0] + boundary_array[1] + boundary_array[2];
    // 111 + 444 + 333 = 888
    
    // Final result: combination of all array operations and indexing patterns
    var final_result = mixed_result + indexed_elem + dyn_elem1 + random_sum + calc_elem + seq_sum + stride_sum + offset_sum + boundary_result;
    // 1031.25 + 20.75 + 84.0 + 24.0 + 6.0 + 60 + 1000 + 1200 + 888 = 4314.0
    
    return final_result;
}