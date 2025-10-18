// SimpLang Tensor Type System Test
// Comprehensive test for all supported tensor datatypes with auto-vectorization

func kernel_main() -> f32 {
    var size = 1048576;  // 1M elements to trigger SIMD optimization
    
    // Test F32 operations (primary floating point)
    var a_f32 = tensor_create_f32(size);
    var b_f32 = tensor_create_f32(size);
    var result_f32 = tensor_create_f32(size);
    
    tensor_fill_f32(a_f32, 2.5, size);
    tensor_fill_f32(b_f32, 1.5, size);
    tensor_add_f32(a_f32, b_f32, result_f32, size);
    var sum_f32 = tensor_sum_f32(result_f32, size);
    
    // Test F64 operations (high precision)
    var a_f64 = tensor_create_f64(size);
    var b_f64 = tensor_create_f64(size);
    var result_f64 = tensor_create_f64(size);
    
    tensor_add_f64(a_f64, b_f64, result_f64, size);
    var sum_f64 = tensor_sum_f64(result_f64, size);
    
    // Test I32 operations (primary integer)
    var a_i32 = tensor_create_i32(size);
    var b_i32 = tensor_create_i32(size);
    var result_i32 = tensor_create_i32(size);
    
    tensor_fill_i32(a_i32, 10, size);
    tensor_fill_i32(b_i32, 5, size);
    tensor_add_i32(a_i32, b_i32, result_i32, size);
    var sum_i32 = tensor_sum_i32(result_i32, size);
    
    // Test I16 operations (compact integers)
    var a_i16 = tensor_create_i16(size);
    var b_i16 = tensor_create_i16(size);
    var result_i16 = tensor_create_i16(size);
    
    tensor_add_i16(a_i16, b_i16, result_i16, size);
    
    // Test U32 operations (unsigned integers)
    var a_u32 = tensor_create_u32(size);
    var b_u32 = tensor_create_u32(size);
    var result_u32 = tensor_create_u32(size);
    
    tensor_add_u32(a_u32, b_u32, result_u32, size);
    
    // Test boolean operations
    var a_bool = tensor_create_bool(size);
    var b_bool = tensor_create_bool(size);
    var result_bool = tensor_create_bool(size);
    
    tensor_and_bool(a_bool, b_bool, result_bool, size);
    
    // Test type conversions
    var converted_f64 = tensor_create_f64(size);
    tensor_f32_to_f64(result_f32, converted_f64, size);
    
    // Verify expected results for F32 test (2.5 + 1.5 = 4.0 per element)
    var expected_f32 = 4.0 * size;
    var success = (sum_f32 == expected_f32);
    
    return sum_f32;  // Return result for verification
}