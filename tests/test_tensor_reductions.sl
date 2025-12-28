// Tensor Reduction Operations Test Suite
// Tests: sum, mean, max, min, argmax
// Dtypes: f32, i32, i64

// Test 1: tensor_sum with f32
fn test_sum_f32() -> f32 {
    f32<2,3> t = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    var sum = tensor_sum(t);
    return sum;  // Expected: 21.0
}

// Test 2: tensor_mean with f32
fn test_mean_f32() -> f32 {
    f32<2,3> t = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    var mean = tensor_mean(t);
    return mean;  // Expected: 3.5 (21/6)
}

// Test 3: tensor_max with f32
fn test_max_f32() -> f32 {
    f32<2,3> t = {1.5, 9.2, 3.1, 4.7, 2.3, 8.6};
    var max_val = tensor_max(t);
    return max_val;  // Expected: 9.2
}

// Test 4: tensor_min with f32
fn test_min_f32() -> f32 {
    f32<2,3> t = {5.5, 2.1, 8.3, 1.2, 9.4, 3.7};
    var min_val = tensor_min(t);
    return min_val;  // Expected: 1.2
}

// Test 5: tensor_argmax with f32
fn test_argmax_f32() -> i64 {
    f32<2,3> t = {1.0, 2.0, 9.5, 4.0, 5.0, 6.0};
    var idx = tensor_argmax(t);
    return idx;  // Expected: 2 (index of 9.5)
}

// Test 6: tensor_sum with i32
fn test_sum_i32() -> i32 {
    i32<3,2> t = {10i, 20i, 30i, 40i, 50i, 60i};
    var sum = tensor_sum(t);
    return sum;  // Expected: 210
}

// Test 7: tensor_max with i32
fn test_max_i32() -> i32 {
    i32<2,2> t = {15i, 42i, 8i, 33i};
    var max_val = tensor_max(t);
    return max_val;  // Expected: 42
}

// Test 8: tensor_min with i32
fn test_min_i32() -> i32 {
    i32<2,2> t = {15i, 42i, 8i, 33i};
    var min_val = tensor_min(t);
    return min_val;  // Expected: 8
}

// Test 9: tensor_argmax with i32
fn test_argmax_i32() -> i64 {
    i32<2,3> t = {10i, 20i, 15i, 25i, 99i, 30i};
    var idx = tensor_argmax(t);
    return idx;  // Expected: 4 (index of 99)
}

// Test 10: tensor_sum with i64
fn test_sum_i64() -> i64 {
    i64<2,2> t = {100, 200, 300, 400};
    var sum = tensor_sum(t);
    return sum;  // Expected: 1000
}

// Test 11: tensor_max with i64
fn test_max_i64() -> i64 {
    i64<3,2> t = {50, 120, 80, 200, 150, 90};
    var max_val = tensor_max(t);
    return max_val;  // Expected: 200
}

// Test 12: tensor_min with i64
fn test_min_i64() -> i64 {
    i64<3,2> t = {50, 120, 80, 200, 150, 90};
    var min_val = tensor_min(t);
    return min_val;  // Expected: 50
}

// Test 13: Multi-dimensional tensor sum (3D)
fn test_sum_3d() -> f32 {
    f32<2,2,2> t = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    var sum = tensor_sum(t);
    return sum;  // Expected: 36.0
}

// Test 14: Single element tensor
fn test_single_element() -> f32 {
    f32<1,1> t = {42.5};
    var sum = tensor_sum(t);
    var max_val = tensor_max(t);
    var min_val = tensor_min(t);
    return sum + max_val - min_val;  // Expected: 42.5 + 42.5 - 42.5 = 42.5
}

// Main test function
fn test_reductions_main() -> f32 {
    var r1 = test_sum_f32();           // 21.0
    var r2 = test_mean_f32();          // 3.5
    var r3 = test_max_f32();           // 9.2
    var r4 = test_min_f32();           // 1.2
    var r5 = test_argmax_f32();        // 2
    var r6 = test_sum_i32();           // 210
    var r7 = test_max_i32();           // 42
    var r8 = test_min_i32();           // 8
    var r9 = test_argmax_i32();        // 4
    var r10 = test_sum_i64();          // 1000
    var r11 = test_max_i64();          // 200
    var r12 = test_min_i64();          // 50
    var r13 = test_sum_3d();           // 36.0
    var r14 = test_single_element();   // 42.5

    // Sum all test results for validation
    // Expected total: 21.0 + 3.5 + 9.2 + 1.2 + 2 + 210 + 42 + 8 + 4 + 1000 + 200 + 50 + 36.0 + 42.5 = 1629.4
    var total = r1 + r2 + r3 + r4;
    total = total + r5 + r6 + r7 + r8;
    total = total + r9 + r10 + r11 + r12;
    total = total + r13 + r14;

    return total;
}
