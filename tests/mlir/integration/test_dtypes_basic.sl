// Test basic dtype support for arrays with C++ style type promotion
// Tests F32, F64, I32, I64 arrays with automatic type conversions in arithmetic

fn test_f32_array() -> f32 {
    var arr_f32 = array<f32>([10]);
    arr_f32[5i] = 3.14;
    return arr_f32[5i];  // Returns f32
}

fn test_f64_array() -> f64 {
    var arr_f64 = array<f64>([10]);
    arr_f64[7i] = 2.718;
    return arr_f64[7i];  // Returns f64
}

fn test_i32_array() -> i32 {
    var arr_i32 = array<i32>([10]);
    arr_i32[3i] = 42;
    return arr_i32[3i];  // Returns i32
}

fn test_i64_array() -> i64 {
    var arr_i64 = array<i64>([10]);
    arr_i64[8i] = 9999;
    return arr_i64[8i];  // Returns i64
}

// Test with multi-dimensional i32 array
fn test_i32_2d() -> i32 {
    var arr2d = array<i32>([4, 5]);  // 4x5 = 20 elements
    arr2d[2i, 3i] = 777;  // flattened: 2*5 + 3 = 13
    return arr2d[2i, 3i];  // Returns i32
}

// Main test - demonstrates C++ style type promotion
// f32 + f64 → f64, i32 + f64 → f64, etc.
fn kernel_main() -> f64 {
    var r1 = test_f32_array();    // f32: 3.14
    var r2 = test_f64_array();    // f64: 2.718
    var r3 = test_i32_array();    // i32: 42
    var r4 = test_i64_array();    // i64: 9999
    var r5 = test_i32_2d();       // i32: 777

    // Type promotion happens automatically:
    // r1 (f32) + r2 (f64) → f64 (promotes r1 to f64)
    // result (f64) + r3 (i32) → f64 (promotes r3 to f64)
    // etc.
    // Expected: 3.14 + 2.718 + 42 + 9999 + 777 = 10823.858
    return r1 + r2 + r3 + r4 + r5;
}
