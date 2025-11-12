// Test axis reductions for tensor operations

// 2D tests - tensor_sum
fn test_sum_2d_axis0_f32() -> f32 {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // Sum along axis 0 (rows) -> [5, 7, 9]
    f32<2,3> t = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    f32<3> result = tensor_sum(t, 0);
    // Return sum of result for validation
    return result[0i] + result[1i] + result[2i];  // Expected: 21.0
}

fn test_sum_2d_axis1_f32() -> f32 {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // Sum along axis 1 (cols) -> [6, 15]
    f32<2,3> t = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    f32<2> result = tensor_sum(t, 1);
    return result[0i] + result[1i];  // Expected: 21.0
}

// 2D tests - tensor_mean
fn test_mean_2d_axis0_f32() -> f32 {
    // [[2, 4, 6],
    //  [8, 10, 12]]
    // Mean along axis 0 -> [5, 7, 9]
    f32<2,3> t = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
    f32<3> result = tensor_mean(t, 0);
    return result[0i] + result[1i] + result[2i];  // Expected: 21.0
}

fn test_mean_2d_axis1_f32() -> f32 {
    // [[3, 3, 3],
    //  [6, 6, 6]]
    // Mean along axis 1 -> [3, 6]
    f32<2,3> t = {3.0, 3.0, 3.0, 6.0, 6.0, 6.0};
    f32<2> result = tensor_mean(t, 1);
    return result[0i] + result[1i];  // Expected: 9.0
}

// 2D tests - tensor_max
fn test_max_2d_axis0_f32() -> f32 {
    // [[1, 9, 3],
    //  [4, 2, 8]]
    // Max along axis 0 -> [4, 9, 8]
    f32<2,3> t = {1.0, 9.0, 3.0, 4.0, 2.0, 8.0};
    f32<3> result = tensor_max(t, 0);
    return result[0i] + result[1i] + result[2i];  // Expected: 21.0
}

fn test_max_2d_axis1_f32() -> f32 {
    // [[1, 5, 3],
    //  [2, 4, 9]]
    // Max along axis 1 -> [5, 9]
    f32<2,3> t = {1.0, 5.0, 3.0, 2.0, 4.0, 9.0};
    f32<2> result = tensor_max(t, 1);
    return result[0i] + result[1i];  // Expected: 14.0
}

// 2D tests - tensor_min
fn test_min_2d_axis0_f32() -> f32 {
    // [[5, 2, 8],
    //  [3, 7, 1]]
    // Min along axis 0 -> [3, 2, 1]
    f32<2,3> t = {5.0, 2.0, 8.0, 3.0, 7.0, 1.0};
    f32<3> result = tensor_min(t, 0);
    return result[0i] + result[1i] + result[2i];  // Expected: 6.0
}

fn test_min_2d_axis1_f32() -> f32 {
    // [[7, 3, 9],
    //  [5, 2, 8]]
    // Min along axis 1 -> [3, 2]
    f32<2,3> t = {7.0, 3.0, 9.0, 5.0, 2.0, 8.0};
    f32<2> result = tensor_min(t, 1);
    return result[0i] + result[1i];  // Expected: 5.0
}

// 2D tests - tensor_argmax
fn test_argmax_2d_axis0_f32() -> i64 {
    // [[1, 9, 3],
    //  [4, 2, 8]]
    // Argmax along axis 0 -> [1, 0, 1] (indices of max in each column)
    f32<2,3> t = {1.0, 9.0, 3.0, 4.0, 2.0, 8.0};
    i64<3> result = tensor_argmax(t, 0);
    return result[0i] + result[1i] + result[2i];  // Expected: 2 (1+0+1)
}

fn test_argmax_2d_axis1_f32() -> i64 {
    // [[1, 9, 3],
    //  [2, 4, 8]]
    // Argmax along axis 1 -> [1, 2] (indices of max in each row)
    f32<2,3> t = {1.0, 9.0, 3.0, 2.0, 4.0, 8.0};
    i64<2> result = tensor_argmax(t, 1);
    return result[0i] + result[1i];  // Expected: 3 (1+2)
}

// 3D tests - tensor_sum
fn test_sum_3d_axis0_f32() -> f32 {
    // 2x2x3 tensor, sum along axis 0
    f32<2,2,3> t = {
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0
    };
    f32<2,3> result = tensor_sum(t, 0);
    var total = 0.0;
    total = total + result[0i,0i] + result[0i,1i] + result[0i,2i];
    total = total + result[1i,0i] + result[1i,1i] + result[1i,2i];
    return total;  // Expected: 78.0 (sum of all)
}

fn test_sum_3d_axis1_f32() -> f32 {
    // 2x2x3 tensor, sum along axis 1
    f32<2,2,3> t = {
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0
    };
    f32<2,3> result = tensor_sum(t, 1);
    var total = 0.0;
    total = total + result[0i,0i] + result[0i,1i] + result[0i,2i];
    total = total + result[1i,0i] + result[1i,1i] + result[1i,2i];
    return total;  // Expected: 78.0
}

fn test_sum_3d_axis2_f32() -> f32 {
    // 2x2x3 tensor, sum along axis 2
    f32<2,2,3> t = {
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0
    };
    f32<2,2> result = tensor_sum(t, 2);
    return result[0i,0i] + result[0i,1i] + result[1i,0i] + result[1i,1i];  // Expected: 78.0
}

// Integer tests
fn test_sum_2d_axis0_i32() -> i32 {
    i32<2,3> t = {10, 20, 30, 40, 50, 60};
    i32<3> result = tensor_sum(t, 0);
    return result[0i] + result[1i] + result[2i];  // Expected: 210
}

fn test_max_2d_axis0_i32() -> i32 {
    i32<2,3> t = {10, 90, 30, 40, 20, 80};
    i32<3> result = tensor_max(t, 0);
    return result[0i] + result[1i] + result[2i];  // Expected: 210
}

// Main test aggregator - float tests only
fn test_axis_reductions_main() -> f32 {
    var r1 = test_sum_2d_axis0_f32();       // 21.0
    var r2 = test_sum_2d_axis1_f32();       // 21.0
    var r3 = test_mean_2d_axis0_f32();      // 21.0
    var r4 = test_mean_2d_axis1_f32();      // 9.0
    var r5 = test_max_2d_axis0_f32();       // 21.0
    var r6 = test_max_2d_axis1_f32();       // 14.0
    var r7 = test_min_2d_axis0_f32();       // 6.0
    var r8 = test_min_2d_axis1_f32();       // 5.0
    var r11 = test_sum_3d_axis0_f32();      // 78.0
    var r12 = test_sum_3d_axis1_f32();      // 78.0
    var r13 = test_sum_3d_axis2_f32();      // 78.0

    return r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r11 + r12 + r13;
    // Expected: 337.0
}
