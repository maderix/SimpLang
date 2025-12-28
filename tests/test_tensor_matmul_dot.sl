// Comprehensive test suite for tensor_matmul and tensor_dot operations

// Test 1: 1D dot product (basic)
fn test_dot_1d() -> f32 {
    f32<10> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    f32<10> b = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

    // Expected: 2*(1+2+...+10) = 2*55 = 110.0
    var result = tensor_dot(a, b);
    return result;
}

// Test 2: 1D dot product with negative numbers
fn test_dot_1d_negative() -> f32 {
    f32<4> a = {1.0, -2.0, 3.0, -4.0};
    f32<4> b = {2.0, 3.0, -1.0, 2.0};

    // Expected: 1*2 + (-2)*3 + 3*(-1) + (-4)*2 = 2 - 6 - 3 - 8 = -15.0
    var result = tensor_dot(a, b);
    return result;
}

// Test 3: 2D matmul - 1x1 edge case
fn test_matmul_1x1() -> f32 {
    f32<1, 1> A = {5.0};
    f32<1, 1> B = {3.0};

    var C = tensor_matmul(A, B);
    return C[0i, 0i];  // Expected: 15.0
}

// Test 4: 2D matmul - 2x2 identity
fn test_matmul_2x2_identity() -> f32 {
    f32<2, 2> A = {
        1.0, 2.0,
        3.0, 4.0
    };

    f32<2, 2> I = {
        1.0, 0.0,
        0.0, 1.0
    };

    var C = tensor_matmul(A, I);
    return C[0i, 0i];  // Expected: 1.0
}

// Test 5: 2D matmul - 3x3 comprehensive
fn test_matmul_3x3() -> f32 {
    f32<3, 3> A = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };

    f32<3, 3> B = {
        9.0, 8.0, 7.0,
        6.0, 5.0, 4.0,
        3.0, 2.0, 1.0
    };

    var C = tensor_matmul(A, B);

    // C[0,0] = 1*9 + 2*6 + 3*3 = 9 + 12 + 9 = 30
    // C[1,1] = 4*8 + 5*5 + 6*2 = 32 + 25 + 12 = 69
    // C[2,2] = 7*7 + 8*4 + 9*1 = 49 + 32 + 9 = 90
    return C[0i, 0i] + C[1i, 1i] + C[2i, 2i];  // Expected: 30 + 69 + 90 = 189.0
}

// Test 6: 2D matmul - 4x4 larger matrix
fn test_matmul_4x4() -> f32 {
    f32<4, 4> A = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0,
        0.0, 0.0, 0.0, 4.0
    };

    f32<4, 4> B = {
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 5.0
    };

    var C = tensor_matmul(A, B);

    // Diagonal matrix multiplication
    // C[0,0]=2, C[1,1]=6, C[2,2]=12, C[3,3]=20
    return C[0i, 0i] + C[1i, 1i] + C[2i, 2i] + C[3i, 3i];  // Expected: 2+6+12+20 = 40.0
}

// Test 7: 2D matmul - non-square (4x3 × 3x5)
fn test_matmul_nonsquare() -> f32 {
    f32<4, 3> A = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    };

    f32<3, 5> B = {
        1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0
    };

    var C = tensor_matmul(A, B);

    // C is 4x5
    // C[0,0] = 1*1 + 2*0 + 3*0 = 1
    // C[0,1] = 1*0 + 2*1 + 3*0 = 2
    // C[0,2] = 1*0 + 2*0 + 3*1 = 3
    // C[3,4] = 10*1 + 11*0 + 12*0 = 10
    return C[0i, 0i] + C[0i, 1i] + C[0i, 2i] + C[3i, 4i];  // Expected: 1+2+3+10 = 16.0
}

// Test 8: 2D matmul - standard GEMM (3x2 × 2x3)
fn test_matmul_2d() -> f32 {
    f32<3, 2> A = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };

    f32<2, 3> B = {
        1.0, 0.0, 2.0,
        0.0, 1.0, 3.0
    };

    var C = tensor_matmul(A, B);
    return C[0i, 0i] + C[0i, 1i] + C[0i, 2i];  // Expected: 1 + 2 + 8 = 11.0
}

// Test 9: 3D batched - 2 batches of 2x3 × 3x2
fn test_matmul_3d_batched() -> f32 {
    f32<2, 2, 3> A = {
        // Batch 0
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        // Batch 1
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    };

    f32<2, 3, 2> B = {
        // Batch 0
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        // Batch 1
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0
    };

    var C = tensor_matmul(A, B);

    // Batch 0: C[0,0,0] = 1*1 + 2*0 + 3*1 = 4
    // Batch 1: C[1,0,0] = 7*1 + 8*0 + 9*1 = 16
    return C[0i, 0i, 0i] + C[1i, 0i, 0i];  // Expected: 20.0
}

// Test 10: 3D batched - larger batches (3 batches of 3x3 × 3x3)
fn test_matmul_3d_large_batch() -> f32 {
    f32<3, 3, 3> A = {
        // Batch 0
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        // Batch 1
        2.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 2.0,
        // Batch 2
        3.0, 0.0, 0.0,
        0.0, 3.0, 0.0,
        0.0, 0.0, 3.0
    };

    f32<3, 3, 3> B = {
        // Batch 0
        5.0, 0.0, 0.0,
        0.0, 5.0, 0.0,
        0.0, 0.0, 5.0,
        // Batch 1
        4.0, 0.0, 0.0,
        0.0, 4.0, 0.0,
        0.0, 0.0, 4.0,
        // Batch 2
        2.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 2.0
    };

    var C = tensor_matmul(A, B);

    // Batch 0: C[0,0,0] = 1*5 = 5
    // Batch 1: C[1,1,1] = 2*4 = 8
    // Batch 2: C[2,2,2] = 3*2 = 6
    return C[0i, 0i, 0i] + C[1i, 1i, 1i] + C[2i, 2i, 2i];  // Expected: 5+8+6 = 19.0
}

// Test 11: 4D NHWC - small fully connected layer
fn test_matmul_4d_nhwc() -> f32 {
    f32<2, 2, 2, 3> input = {
        // N=0, H=0, W=0
        1.0, 2.0, 3.0,
        // N=0, H=0, W=1
        4.0, 5.0, 6.0,
        // N=0, H=1, W=0
        7.0, 8.0, 9.0,
        // N=0, H=1, W=1
        10.0, 11.0, 12.0,
        // N=1, H=0, W=0
        13.0, 14.0, 15.0,
        // N=1, H=0, W=1
        16.0, 17.0, 18.0,
        // N=1, H=1, W=0
        19.0, 20.0, 21.0,
        // N=1, H=1, W=1
        22.0, 23.0, 24.0
    };

    f32<4, 3> weights = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.5, 0.5, 0.5
    };

    var output = tensor_matmul(input, weights);

    // output[0,0,0,0] = [1,2,3] × [1,0,0]^T = 1
    // output[0,0,0,3] = [1,2,3] × [0.5,0.5,0.5]^T = 3.0
    return output[0i, 0i, 0i, 0i] + output[0i, 0i, 0i, 3i];  // Expected: 4.0
}

// Test 12: 4D NHWC - larger spatial dimensions (3x4x4x2 × 3x2)
fn test_matmul_4d_nhwc_large() -> f32 {
    f32<2, 3, 3, 2> input = {
        // N=0
        1.0, 1.0,  2.0, 2.0,  3.0, 3.0,
        4.0, 4.0,  5.0, 5.0,  6.0, 6.0,
        7.0, 7.0,  8.0, 8.0,  9.0, 9.0,
        // N=1
        10.0, 10.0,  11.0, 11.0,  12.0, 12.0,
        13.0, 13.0,  14.0, 14.0,  15.0, 15.0,
        16.0, 16.0,  17.0, 17.0,  18.0, 18.0
    };

    f32<3, 2> weights = {
        1.0, 1.0,
        2.0, 2.0,
        3.0, 3.0
    };

    var output = tensor_matmul(input, weights);

    // output[0,0,0,0] = [1,1] × [1,1]^T = 1+1 = 2
    // output[0,0,0,1] = [1,1] × [2,2]^T = 2+2 = 4
    // output[0,0,0,2] = [1,1] × [3,3]^T = 3+3 = 6
    return output[0i, 0i, 0i, 0i] + output[0i, 0i, 0i, 1i] + output[0i, 0i, 0i, 2i];  // Expected: 2+4+6 = 12.0
}

// Test 13: Verify all elements of a simple 2x2 matmul
fn test_matmul_2x2_full() -> f32 {
    f32<2, 2> A = {
        1.0, 2.0,
        3.0, 4.0
    };

    f32<2, 2> B = {
        5.0, 6.0,
        7.0, 8.0
    };

    var C = tensor_matmul(A, B);

    // C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
    // C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
    // C[1,0] = 3*5 + 4*7 = 15 + 28 = 43
    // C[1,1] = 3*6 + 4*8 = 18 + 32 = 50
    return C[0i, 0i] + C[0i, 1i] + C[1i, 0i] + C[1i, 1i];  // Expected: 19+22+43+50 = 134.0
}

// Test 14: Zero matrix multiplication
fn test_matmul_zeros() -> f32 {
    f32<2, 2> A = {
        1.0, 2.0,
        3.0, 4.0
    };

    f32<2, 2> Z = {
        0.0, 0.0,
        0.0, 0.0
    };

    var C = tensor_matmul(A, Z);

    // All elements should be 0
    return C[0i, 0i] + C[0i, 1i] + C[1i, 0i] + C[1i, 1i];  // Expected: 0.0
}

// Test 15: Larger dot product (100 elements)
fn test_dot_large() -> f32 {
    f32<100> a;
    f32<100> b;

    // Initialize using memory - all 1.0
    var i = 0.0;
    while (i < 100.0) {
        var idx = i as i64;
        a[idx] = 1.0;
        b[idx] = 1.0;
        i = i + 1.0;
    }

    var result = tensor_dot(a, b);
    return result;  // Expected: 100.0
}

// Main comprehensive test runner
fn kernel_main() -> f32 {
    var sum = 0.0;

    sum = sum + test_dot_1d();                    // 110.0
    sum = sum + test_dot_1d_negative();           // -15.0
    sum = sum + test_matmul_1x1();                // 15.0
    sum = sum + test_matmul_2x2_identity();       // 1.0
    sum = sum + test_matmul_3x3();                // 189.0
    sum = sum + test_matmul_4x4();                // 40.0
    sum = sum + test_matmul_nonsquare();          // 16.0
    sum = sum + test_matmul_2d();                 // 11.0
    sum = sum + test_matmul_3d_batched();         // 20.0
    sum = sum + test_matmul_3d_large_batch();     // 19.0
    sum = sum + test_matmul_4d_nhwc();            // 4.0
    sum = sum + test_matmul_4d_nhwc_large();      // 12.0
    sum = sum + test_matmul_2x2_full();           // 134.0
    sum = sum + test_matmul_zeros();              // 0.0
    sum = sum + test_dot_large();                 // 100.0

    // Total expected: 110 - 15 + 15 + 1 + 189 + 40 + 16 + 11 + 20 + 19 + 4 + 12 + 134 + 0 + 100 = 656.0
    return sum;
}
