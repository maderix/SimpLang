// Test basic static types without mixing different types

fn test_f32_only() -> f32 {
    f32 a = 1.5;
    f32 b = 2.5;
    f32 result = a + b;
    return result;
}

fn test_i32_only() -> i32 {
    i32 x = 10;
    i32 y = 20;
    i32 sum = x + y;
    return sum;
}

fn test_f64_only() -> f64 {
    f64 pi = 3.14159;
    f64 e = 2.71828;
    f64 combined = pi + e;
    return combined;
}

fn kernel_main() {
    // Test each type in isolation
    f32 f32_result = test_f32_only();
    i32 i32_result = test_i32_only();
    f64 f64_result = test_f64_only();
    
    // Simple addition with consistent f32 types
    f32 simple_sum = f32_result + 1.0;
    
    // Convert to double for kernel_main return (SimpLang requirement)
    f64 final_result = simple_sum;
    return final_result;
}