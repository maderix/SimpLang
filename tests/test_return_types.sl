// Test explicit return types

fn test_f32_return() -> f32 {
    f32 result = 3.5;
    return result;
}

fn test_i32_return() -> i32 {
    i32 result = 42;
    return result;
}

fn test_f64_return() -> f64 {
    f64 result = 3.14159;
    return result;
}

fn kernel_main() {
    // Call functions with explicit return types
    f32 f32_val = test_f32_return();
    i32 i32_val = test_i32_return();
    f64 f64_val = test_f64_return();
    
    // Simple return (SimpLang still needs the main to return double)
    return 1.0;
}