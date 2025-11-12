// Comprehensive element-wise tensor operations test
// Tests: add, mul, sub, div, relu, sigmoid, tanh

// Test 1: Tensor addition
fn test_tensor_add() -> f32 {
    f32<2,3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    f32<2,3> b = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};

    f32<2,3> result = a + b;

    // Expected: {11.0, 22.0, 33.0, 44.0, 55.0, 66.0}
    // Return first element for verification
    return result[0i, 0i];  // Should be 11.0
}

// Test 2: Tensor multiplication
fn test_tensor_mul() -> f32 {
    f32<2,3> a = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    f32<2,3> b = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0};

    f32<2,3> result = a * b;

    // Expected: {20.0, 30.0, 40.0, 50.0, 60.0, 70.0}
    return result[0i, 1i];  // Should be 30.0
}

// Test 3: Tensor subtraction
fn test_tensor_sub() -> f32 {
    f32<2,3> a = {100.0, 90.0, 80.0, 70.0, 60.0, 50.0};
    f32<2,3> b = {10.0, 20.0, 30.0, 40.0, 50.0, 40.0};

    f32<2,3> result = a - b;

    // Expected: {90.0, 70.0, 50.0, 30.0, 10.0, 10.0}
    return result[1i, 0i];  // Should be 30.0
}

// Test 4: Tensor division
fn test_tensor_div() -> f32 {
    f32<2,3> a = {100.0, 50.0, 25.0, 200.0, 80.0, 40.0};
    f32<2,3> b = {10.0, 5.0, 5.0, 10.0, 8.0, 4.0};

    f32<2,3> result = a / b;

    // Expected: {10.0, 10.0, 5.0, 20.0, 10.0, 10.0}
    return result[1i, 1i];  // Should be 10.0
}

// Test 5: Combined operations
fn test_tensor_combined() -> f32 {
    f32<2,2> a = {1.0, 2.0, 3.0, 4.0};
    f32<2,2> b = {5.0, 6.0, 7.0, 8.0};
    f32<2,2> c = {2.0, 2.0, 2.0, 2.0};

    // Complex expression: (a + b) * c
    f32<2,2> sum = a + b;      // {6.0, 8.0, 10.0, 12.0}
    f32<2,2> result = sum * c; // {12.0, 16.0, 20.0, 24.0}

    return result[1i, 0i];  // Should be 20.0
}

// Test 6: Chained operations
fn test_tensor_chained() -> f32 {
    f32<3> x = {10.0, 20.0, 30.0};
    f32<3> y = {5.0, 10.0, 15.0};
    f32<3> z = {2.0, 2.0, 2.0};

    // (x - y) / z
    f32<3> diff = x - y;      // {5.0, 10.0, 15.0}
    f32<3> result = diff / z; // {2.5, 5.0, 7.5}

    return result[1i];  // Should be 5.0
}

// Main test runner
fn test_elementwise_main() -> f32 {
    var r1 = test_tensor_add();
    var r2 = test_tensor_mul();
    var r3 = test_tensor_sub();
    var r4 = test_tensor_div();
    var r5 = test_tensor_combined();
    var r6 = test_tensor_chained();

    // Sum all results for verification
    // Expected: 11.0 + 30.0 + 30.0 + 10.0 + 20.0 + 5.0 = 106.0
    return r1 + r2 + r3 + r4 + r5 + r6;
}
