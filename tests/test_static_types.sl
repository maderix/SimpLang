fn test_basic_types() {
    // Basic static types
    f32 weight = 1.5;
    f64 precise = 3.14159265359;
    i32 count = 42;
    i64 big_num = 1000000;
    bool flag = 1.0;  // SimpLang uses double for booleans currently
    
    // Mixed with dynamic
    var dynamic_var = weight + precise;
    
    return weight + precise + count + big_num + flag + dynamic_var;
}

fn test_arrays() {
    // Static arrays
    f32[4] weights;
    i32[3] dimensions;
    
    // Array assignments would need array syntax extensions
    // For now just return placeholder
    return 1.0;
}

fn test_function_params(f32 input, i32 size) {
    // Function with static typed parameters
    f32 result = input * 2.0;
    i32 doubled_size = size + size;  // Note: would need int arithmetic in codegen
    
    return result;  // Return f32 as double for now
}

fn kernel_main() {
    // Test basic static types
    f32 basic_result = test_basic_types();
    
    // Test arrays 
    f32 array_result = test_arrays();
    
    // Test function parameters
    f32 param_result = test_function_params(5.5, 10);
    
    return basic_result + array_result + param_result;
}