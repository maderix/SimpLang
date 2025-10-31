// High-dimensional tensor tests (4D, 5D)
// Verify basic infrastructure works at higher dimensions

// Test 4D tensor - basic operations
fn test_4d_basic() -> f32 {
    // 2x2x2x2 = 16 elements
    f32<2,2,2,2> t = {
        1.0, 2.0,   3.0, 4.0,
        5.0, 6.0,   7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };

    // Access elements
    var v1 = t[0i,0i,0i,0i];  // 1.0
    var v2 = t[1i,1i,1i,1i];  // 16.0
    var v3 = t[0i,1i,0i,1i];  // 6.0 (index [0,1,0,1] = 0*8 + 1*4 + 0*2 + 1 = 5 → element 5)

    return v1 + v2 + v3;  // 1 + 16 + 6 = 23
}

// Test 4D tensor - set and get
fn test_4d_set_get() -> f32 {
    f32<2,2,2,2> t;

    t[0i,0i,0i,0i] = 5.0;
    t[1i,0i,1i,0i] = 10.0;
    t[1i,1i,1i,1i] = 15.0;

    var sum = t[0i,0i,0i,0i] + t[1i,0i,1i,0i] + t[1i,1i,1i,1i];
    return sum;  // 5 + 10 + 15 = 30
}

// Test 4D tensor - element-wise operations
fn test_4d_elementwise() -> f32 {
    f32<2,2,2,2> a = {
        1.0, 2.0,  3.0, 4.0,
        5.0, 6.0,  7.0, 8.0,
        1.0, 2.0,  3.0, 4.0,
        5.0, 6.0,  7.0, 8.0
    };

    f32<2,2,2,2> b = {
        2.0, 2.0,  2.0, 2.0,
        2.0, 2.0,  2.0, 2.0,
        2.0, 2.0,  2.0, 2.0,
        2.0, 2.0,  2.0, 2.0
    };

    f32<2,2,2,2> c = a + b;

    // Sum should be all elements * 2 + 2
    // a sums to 72, b sums to 32, c should sum to 104
    var result = c[0i,0i,0i,0i] + c[1i,1i,1i,1i];  // 3 + 10 = 13
    return result;
}

// Test 4D tensor - reduction
fn test_4d_reduction() -> f32 {
    f32<2,2,2,2> t = {
        1.0, 1.0,  1.0, 1.0,
        1.0, 1.0,  1.0, 1.0,
        1.0, 1.0,  1.0, 1.0,
        1.0, 1.0,  1.0, 1.0
    };

    var sum = tensor_sum(t);  // Should be 16.0
    return sum;
}

// Test 5D tensor - basic operations
fn test_5d_basic() -> f32 {
    // 2x2x2x2x2 = 32 elements (keeping small for test)
    f32<2,2,2,2,2> t = {
        // First 4D slice [0,:,:,:,:]
        1.0, 2.0,   3.0, 4.0,
        5.0, 6.0,   7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        // Second 4D slice [1,:,:,:,:]
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0,
        29.0, 30.0, 31.0, 32.0
    };

    var v1 = t[0i,0i,0i,0i,0i];  // 1.0
    var v2 = t[1i,1i,1i,1i,1i];  // 32.0
    var v3 = t[1i,0i,0i,0i,0i];  // 17.0

    return v1 + v2 + v3;  // 1 + 32 + 17 = 50
}

// Test 5D tensor - set and get
fn test_5d_set_get() -> f32 {
    f32<2,2,2,2,2> t;

    t[0i,0i,0i,0i,0i] = 100.0;
    t[1i,1i,1i,1i,1i] = 200.0;
    t[0i,1i,0i,1i,0i] = 50.0;

    var sum = t[0i,0i,0i,0i,0i] + t[1i,1i,1i,1i,1i] + t[0i,1i,0i,1i,0i];
    return sum;  // 100 + 200 + 50 = 350
}

// Test 5D tensor - reduction
fn test_5d_reduction() -> f32 {
    f32<2,2,2,2,2> t = {
        1.0, 1.0,  1.0, 1.0,
        1.0, 1.0,  1.0, 1.0,
        1.0, 1.0,  1.0, 1.0,
        1.0, 1.0,  1.0, 1.0,
        1.0, 1.0,  1.0, 1.0,
        1.0, 1.0,  1.0, 1.0,
        1.0, 1.0,  1.0, 1.0,
        1.0, 1.0,  1.0, 1.0
    };

    var sum = tensor_sum(t);  // Should be 32.0
    return sum;
}

// Main aggregator
fn test_highd_main() -> f32 {
    var r1 = test_4d_basic();        // 23
    var r2 = test_4d_set_get();      // 30
    var r3 = test_4d_elementwise();  // 13
    var r4 = test_4d_reduction();    // 16
    var r5 = test_5d_basic();        // 50
    var r6 = test_5d_set_get();      // 350
    var r7 = test_5d_reduction();    // 32

    return r1 + r2 + r3 + r4 + r5 + r6 + r7;
    // Expected: 23 + 30 + 13 + 16 + 50 + 350 + 32 = 514
}
