// Memory operations tests: reshape, transpose, slice

// Test 1: Reshape 1D to 2D
fn test_reshape_1d_to_2d() -> f32 {
    f32<6> flat = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    f32<2,3> reshaped = tensor_reshape(flat, 2, 3);
    // [[1, 2, 3],
    //  [4, 5, 6]]
    return reshaped[0i,0i] + reshaped[1i,2i];  // 1 + 6 = 7
}

// Test 2: Reshape 2D to 1D
fn test_reshape_2d_to_1d() -> f32 {
    f32<2,3> mat = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    f32<6> flat = tensor_reshape(mat, 6);
    return flat[0i] + flat[5i];  // 10 + 60 = 70
}

// Test 3: Reshape 2D to different 2D
fn test_reshape_2d_to_2d() -> f32 {
    f32<2,3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    f32<3,2> b = tensor_reshape(a, 3, 2);
    // [[1, 2],
    //  [3, 4],
    //  [5, 6]]
    return b[0i,0i] + b[2i,1i];  // 1 + 6 = 7
}

// Test 4: Transpose 2D (default)
fn test_transpose_2d() -> f32 {
    f32<2,3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    // [[1, 2, 3],
    //  [4, 5, 6]]
    f32<3,2> b = tensor_transpose(a);
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    return b[0i,0i] + b[2i,1i];  // 1 + 6 = 7
}

// Test 5: Transpose 3D with permutation
fn test_transpose_3d() -> f32 {
    f32<2,3,4> a = {
        1.0, 2.0, 3.0, 4.0,    5.0, 6.0, 7.0, 8.0,    9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
    };
    // Shape <2,3,4> permuted [2,0,1] -> <4,2,3>
    f32<4,2,3> b = tensor_transpose(a, 2, 0, 1);
    return b[0i,0i,0i] + b[3i,1i,2i];  // 1 + 24 = 25
}

// Test 6: Slice 2D tensor
fn test_slice_2d() -> f32 {
    f32<4,5> a = {
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0
    };
    // Slice [1:3, 2:4] -> [[8, 9], [13, 14]]
    f32<2,2> b = tensor_slice(a, 1, 3, 2, 4);
    return b[0i,0i] + b[1i,1i];  // 8 + 14 = 22
}

// Test 7: Slice 3D tensor
fn test_slice_3d() -> f32 {
    f32<2,3,4> a = {
        1.0, 2.0, 3.0, 4.0,    5.0, 6.0, 7.0, 8.0,    9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
    };
    // Slice [0:1, 1:3, 2:4]
    f32<1,2,2> b = tensor_slice(a, 0, 1, 1, 3, 2, 4);
    return b[0i,0i,0i] + b[0i,1i,1i];  // 7 + 12 = 19
}

// Test 8: Combined operations
fn test_combined_ops() -> f32 {
    // Reshape -> Transpose -> Slice
    f32<6> flat = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Reshape to 2x3
    f32<2,3> mat = tensor_reshape(flat, 2, 3);
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Transpose to 3x2
    f32<3,2> trans = tensor_transpose(mat);
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]

    // Slice [1:3, 0:2]
    f32<2,2> sliced = tensor_slice(trans, 1, 3, 0, 2);
    // [[2, 5],
    //  [3, 6]]

    return sliced[0i,0i] + sliced[1i,1i];  // 2 + 6 = 8
}

// Test 9: Reshape 4D
fn test_reshape_4d() -> f32 {
    f32<2,2,2,2> a = {
        1.0, 2.0,  3.0, 4.0,
        5.0, 6.0,  7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };
    // 2x2x2x2 = 16 elements -> reshape to 4x4
    f32<4,4> b = tensor_reshape(a, 4, 4);
    return b[0i,0i] + b[3i,3i];  // 1 + 16 = 17
}

// Test 10: Transpose 4D
fn test_transpose_4d() -> f32 {
    f32<2,2,2,2> a = {
        1.0, 2.0,  3.0, 4.0,
        5.0, 6.0,  7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };
    // Permute [3,2,1,0] -> reverse all dimensions
    f32<2,2,2,2> b = tensor_transpose(a, 3, 2, 1, 0);
    return b[0i,0i,0i,0i] + b[1i,1i,1i,1i];  // 1 + 16 = 17
}

// Test 11: Slice 4D
fn test_slice_4d() -> f32 {
    f32<3,3,3,3> a = {
        1.0, 2.0, 3.0,  4.0, 5.0, 6.0,  7.0, 8.0, 9.0,
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,

        28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
        37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0,
        46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0,

        55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
        64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0,
        73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0
    };
    // Slice [1:2, 1:2, 1:2, 1:2] -> single element at [1,1,1,1]
    f32<1,1,1,1> b = tensor_slice(a, 1, 2, 1, 2, 1, 2, 1, 2);
    return b[0i,0i,0i,0i];  // 41
}

// Test 12: Reshape 5D
fn test_reshape_5d() -> f32 {
    f32<2,2,2,2,2> a = {
        1.0, 2.0,  3.0, 4.0,
        5.0, 6.0,  7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0,
        29.0, 30.0, 31.0, 32.0
    };
    // 2x2x2x2x2 = 32 elements -> reshape to 4x8
    f32<4,8> b = tensor_reshape(a, 4, 8);
    return b[0i,0i] + b[3i,7i];  // 1 + 32 = 33
}

// Test 13: Transpose 5D
fn test_transpose_5d() -> f32 {
    f32<2,2,2,2,2> a = {
        1.0, 2.0,  3.0, 4.0,
        5.0, 6.0,  7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0,
        29.0, 30.0, 31.0, 32.0
    };
    // Permute [4,3,2,1,0] -> reverse all dimensions
    f32<2,2,2,2,2> b = tensor_transpose(a, 4, 3, 2, 1, 0);
    return b[0i,0i,0i,0i,0i] + b[1i,1i,1i,1i,1i];  // 1 + 32 = 33
}

// Test 14: Slice 5D
fn test_slice_5d() -> f32 {
    f32<2,2,2,2,2> a = {
        1.0, 2.0,  3.0, 4.0,
        5.0, 6.0,  7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0,
        29.0, 30.0, 31.0, 32.0
    };
    // Slice [0:1, 0:1, 0:1, 0:1, 0:1] -> first element
    f32<1,1,1,1,1> b = tensor_slice(a, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    return b[0i,0i,0i,0i,0i];  // 1
}

// Main aggregator
fn test_memory_ops_main() -> f32 {
    var r1 = test_reshape_1d_to_2d();  // 7
    var r2 = test_reshape_2d_to_1d();  // 70
    var r3 = test_reshape_2d_to_2d();  // 7
    var r4 = test_transpose_2d();      // 7
    var r5 = test_transpose_3d();      // 25
    var r6 = test_slice_2d();          // 22
    var r7 = test_slice_3d();          // 19
    var r8 = test_combined_ops();      // 8
    var r9 = test_reshape_4d();        // 17
    var r10 = test_transpose_4d();     // 17
    var r11 = test_slice_4d();         // 41
    var r12 = test_reshape_5d();       // 33
    var r13 = test_transpose_5d();     // 33
    var r14 = test_slice_5d();         // 1

    return r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14;
    // Expected: 7 + 70 + 7 + 7 + 25 + 22 + 19 + 8 + 17 + 17 + 41 + 33 + 33 + 1 = 307
}
