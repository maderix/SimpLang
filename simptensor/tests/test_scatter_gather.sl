// Comprehensive scatter/gather tests for N-D tensors
// Tests cover 1D, 2D, 3D, 4D, 5D with different axes

// ============================================================================
// GATHER TESTS
// ============================================================================

// Test 1: 1D gather (embedding lookup simulation)
fn test_gather_1d() -> f32 {
    f32<10> source = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    i64<4> indices = {0, 3, 7, 2};
    f32<4> result = tensor_gather(source, indices, 0);
    // result should be {0.0, 3.0, 7.0, 2.0}
    return result[0i] + result[1i] + result[2i] + result[3i];  // 0+3+7+2 = 12.0
}

// Test 2: 1D gather with out-of-order indices
fn test_gather_1d_reversed() -> f32 {
    f32<8> source = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};
    i64<3> indices = {7, 2, 0};
    f32<3> result = tensor_gather(source, indices, 0);
    // result should be {80.0, 30.0, 10.0}
    return result[0i] + result[1i] + result[2i];  // 80+30+10 = 120.0
}

// Test 3: 2D gather along axis 0 (select rows)
fn test_gather_2d_axis0() -> f32 {
    f32<5,3> source = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
        13.0, 14.0, 15.0
    };
    i64<3> indices = {4, 1, 0};
    f32<3,3> result = tensor_gather(source, indices, 0);
    // result should be:
    // [[13.0, 14.0, 15.0],   // row 4
    //  [4.0,  5.0,  6.0],    // row 1
    //  [1.0,  2.0,  3.0]]    // row 0
    return result[0i,0i] + result[1i,1i] + result[2i,2i];  // 13+5+3 = 21.0
}

// Test 4: 2D gather along axis 1 (select columns)
fn test_gather_2d_axis1() -> f32 {
    f32<3,5> source = {
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0
    };
    i64<2> indices = {4, 1};
    f32<3,2> result = tensor_gather(source, indices, 1);
    // result should be:
    // [[5.0, 2.0],    // cols 4,1 from row 0
    //  [10.0, 7.0],   // cols 4,1 from row 1
    //  [15.0, 12.0]]  // cols 4,1 from row 2
    return result[0i,0i] + result[1i,1i] + result[2i,0i];  // 5+7+15 = 27.0
}

// Test 5: 3D gather along axis 0
fn test_gather_3d_axis0() -> f32 {
    f32<4,2,3> source = {
        // Slice 0
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        // Slice 1
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
        // Slice 2
        13.0, 14.0, 15.0,
        16.0, 17.0, 18.0,
        // Slice 3
        19.0, 20.0, 21.0,
        22.0, 23.0, 24.0
    };
    i64<2> indices = {3, 0};
    f32<2,2,3> result = tensor_gather(source, indices, 0);
    // result should select slices 3 and 0
    return result[0i,0i,0i] + result[1i,1i,2i];  // 19+6 = 25.0
}

// Test 6: 3D gather along axis 1
fn test_gather_3d_axis1() -> f32 {
    f32<2,4,3> source = {
        // Slice 0
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
        // Slice 1
        13.0, 14.0, 15.0,
        16.0, 17.0, 18.0,
        19.0, 20.0, 21.0,
        22.0, 23.0, 24.0
    };
    i64<2> indices = {3, 0};
    f32<2,2,3> result = tensor_gather(source, indices, 1);
    // Shape: <2,2,3> - gather rows 3,0 from each of 2 slices
    return result[0i,0i,0i] + result[1i,1i,2i];  // 10+15 = 25.0
}

// Test 7: 3D gather along axis 2
fn test_gather_3d_axis2() -> f32 {
    f32<2,3,5> source = {
        // Slice 0
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        // Slice 1
        16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0,
        26.0, 27.0, 28.0, 29.0, 30.0
    };
    i64<3> indices = {4, 1, 0};
    f32<2,3,3> result = tensor_gather(source, indices, 2);
    // Shape: <2,3,3> - gather cols 4,1,0 from each row of each slice
    return result[0i,0i,0i] + result[1i,2i,1i];  // 5+27 = 32.0
}

// Test 8: 4D gather along axis 2
fn test_gather_4d_axis2() -> f32 {
    // Create a 4D tensor <2,2,3,4>
    f32<2,2,3,4> source = {
        // Outer slice 0
        // Inner slice 0
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        // Inner slice 1
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0,
        // Outer slice 1
        // Inner slice 0
        25.0, 26.0, 27.0, 28.0,
        29.0, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0,
        // Inner slice 1
        37.0, 38.0, 39.0, 40.0,
        41.0, 42.0, 43.0, 44.0,
        45.0, 46.0, 47.0, 48.0
    };
    i64<2> indices = {2, 0};
    f32<2,2,2,4> result = tensor_gather(source, indices, 2);
    // Gather slices 2,0 along axis 2
    return result[0i,0i,0i,0i] + result[1i,1i,1i,3i];  // 9+40 = 49.0
}

// ============================================================================
// SCATTER TESTS
// ============================================================================

// Test 9: 1D scatter (sparse update)
fn test_scatter_1d() -> f32 {
    f32<10> dst = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    i64<4> indices = {1, 5, 8, 3};
    f32<4> values = {10.0, 20.0, 30.0, 40.0};
    f32<10> result = tensor_scatter(dst, indices, values, 0);
    // result[1]=10, result[5]=20, result[8]=30, result[3]=40, rest=0
    return result[1i] + result[5i] + result[8i] + result[3i];  // 10+20+30+40 = 100.0
}

// Test 10: 1D scatter with existing values
fn test_scatter_1d_overwrite() -> f32 {
    f32<8> dst = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    i64<3> indices = {0, 4, 7};
    f32<3> values = {100.0, 200.0, 300.0};
    f32<8> result = tensor_scatter(dst, indices, values, 0);
    // result[0]=100, result[4]=200, result[7]=300, rest unchanged
    return result[0i] + result[4i] + result[7i] + result[2i];  // 100+200+300+3 = 603.0
}

// Test 11: 2D scatter along axis 0 (update rows)
fn test_scatter_2d_axis0() -> f32 {
    f32<5,3> dst = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
        13.0, 14.0, 15.0
    };
    i64<2> indices = {4, 1};
    f32<2,3> values = {
        100.0, 200.0, 300.0,  // new row 4
        40.0, 50.0, 60.0      // new row 1
    };
    f32<5,3> result = tensor_scatter(dst, indices, values, 0);
    // row 0: unchanged (1,2,3)
    // row 1: (40,50,60)
    // row 2: unchanged (7,8,9)
    // row 3: unchanged (10,11,12)
    // row 4: (100,200,300)
    return result[4i,0i] + result[1i,1i] + result[0i,0i];  // 100+50+1 = 151.0
}

// Test 12: 2D scatter along axis 1 (update columns)
fn test_scatter_2d_axis1() -> f32 {
    f32<3,5> dst = {
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0
    };
    i64<2> indices = {4, 0};
    f32<3,2> values = {
        100.0, 10.0,   // new cols 4,0 for row 0
        200.0, 20.0,   // new cols 4,0 for row 1
        300.0, 30.0    // new cols 4,0 for row 2
    };
    f32<3,5> result = tensor_scatter(dst, indices, values, 1);
    // row 0: (10, 2, 3, 4, 100)
    // row 1: (20, 7, 8, 9, 200)
    // row 2: (30, 12, 13, 14, 300)
    return result[0i,4i] + result[1i,0i] + result[2i,2i];  // 100+20+13 = 133.0
}

// Test 13: 3D scatter along axis 0
fn test_scatter_3d_axis0() -> f32 {
    f32<4,2,3> dst = {
        // Slice 0
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        // Slice 1
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
        // Slice 2
        13.0, 14.0, 15.0,
        16.0, 17.0, 18.0,
        // Slice 3
        19.0, 20.0, 21.0,
        22.0, 23.0, 24.0
    };
    i64<2> indices = {3, 0};
    f32<2,2,3> values = {
        // New slice 3
        100.0, 200.0, 300.0,
        400.0, 500.0, 600.0,
        // New slice 0
        10.0, 20.0, 30.0,
        40.0, 50.0, 60.0
    };
    f32<4,2,3> result = tensor_scatter(dst, indices, values, 0);
    return result[3i,0i,0i] + result[0i,1i,2i] + result[1i,0i,0i];  // 100+60+7 = 167.0
}

// Test 14: 3D scatter along axis 2
fn test_scatter_3d_axis2() -> f32 {
    f32<2,2,5> dst = {
        // Slice 0
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        // Slice 1
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0
    };
    i64<2> indices = {4, 0};
    f32<2,2,2> values = {
        // Slice 0
        100.0, 10.0,   // cols 4,0 of row 0
        200.0, 20.0,   // cols 4,0 of row 1
        // Slice 1
        300.0, 30.0,   // cols 4,0 of row 0
        400.0, 40.0    // cols 4,0 of row 1
    };
    f32<2,2,5> result = tensor_scatter(dst, indices, values, 2);
    return result[0i,0i,4i] + result[1i,1i,0i] + result[0i,1i,2i];  // 100+40+8 = 148.0
}

// Test 15: 4D scatter along axis 1
fn test_scatter_4d_axis1() -> f32 {
    f32<2,4,2,3> dst = {
        // Outer 0, Inner 0
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        // Outer 0, Inner 1
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
        // Outer 0, Inner 2
        13.0, 14.0, 15.0,
        16.0, 17.0, 18.0,
        // Outer 0, Inner 3
        19.0, 20.0, 21.0,
        22.0, 23.0, 24.0,
        // Outer 1, Inner 0
        25.0, 26.0, 27.0,
        28.0, 29.0, 30.0,
        // Outer 1, Inner 1
        31.0, 32.0, 33.0,
        34.0, 35.0, 36.0,
        // Outer 1, Inner 2
        37.0, 38.0, 39.0,
        40.0, 41.0, 42.0,
        // Outer 1, Inner 3
        43.0, 44.0, 45.0,
        46.0, 47.0, 48.0
    };
    i64<2> indices = {3, 0};
    f32<2,2,2,3> values = {
        // Outer 0, update indices 3,0 along axis 1
        // New inner 3
        100.0, 200.0, 300.0,
        110.0, 210.0, 310.0,
        // New inner 0
        10.0, 20.0, 30.0,
        11.0, 21.0, 31.0,
        // Outer 1, update indices 3,0 along axis 1
        // New inner 3
        400.0, 500.0, 600.0,
        410.0, 510.0, 610.0,
        // New inner 0
        40.0, 50.0, 60.0,
        41.0, 51.0, 61.0
    };
    f32<2,4,2,3> result = tensor_scatter(dst, indices, values, 1);
    return result[0i,3i,0i,0i] + result[1i,0i,1i,2i] + result[0i,1i,0i,0i];  // 100+61+7 = 168.0
}

// Test 16: Combined gather then scatter
fn test_gather_scatter_combined() -> f32 {
    // Create source data
    f32<6,4> source = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0
    };

    // Gather rows 4, 1, 3
    i64<3> gather_indices = {4, 1, 3};
    f32<3,4> gathered = tensor_gather(source, gather_indices, 0);
    // gathered = [[17,18,19,20], [5,6,7,8], [13,14,15,16]]

    // Scatter gathered data to new positions
    f32<6,4> dst = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };
    i64<3> scatter_indices = {0, 2, 5};
    f32<6,4> result = tensor_scatter(dst, scatter_indices, gathered, 0);
    // result row 0 = [17,18,19,20]
    // result row 2 = [5,6,7,8]
    // result row 5 = [13,14,15,16]

    return result[0i,0i] + result[2i,1i] + result[5i,3i];  // 17+6+16 = 39.0
}
