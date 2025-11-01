// Test suite for tensor_matmul and tensor_dot operations

// Test 1: 1D dot product
fn test_dot_1d() -> f64 {
    // Create two 1D tensors of size 10
    f64<10> a;
    f64<10> b;

    // Initialize with simple values
    var i = 0.0;
    while (i < 10.0) {
        var idx = i as i64;
        tensor_store(a, [idx], i + 1.0);  // a = [1,2,3,...,10]
        tensor_store(b, [idx], 2.0);      // b = [2,2,2,...,2]
        i = i + 1.0;
    }

    // Compute dot product: expected = 1*2 + 2*2 + ... + 10*2 = 2*(1+2+...+10) = 2*55 = 110.0
    var result = tensor_dot(a, b);

    return result;  // Should return 110.0
}

// Test 2: 2D matrix multiplication (standard GEMM)
// A(4x3) × B(3x2) = C(4x2)
fn test_matmul_2d() -> f64 {
    f64<4, 3> A;
    f64<3, 2> B;

    // Initialize A with row-major values [1,2,3; 4,5,6; 7,8,9; 10,11,12]
    var i = 0.0;
    while (i < 4.0) {
        var j = 0.0;
        while (j < 3.0) {
            var val = (i * 3.0 + j) + 1.0;
            var ii = i as i64;
            var jj = j as i64;
            tensor_store(A, [ii, jj], val);
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    // Initialize B  [1,2; 3,4; 5,6]
    i = 0.0;
    while (i < 3.0) {
        var j = 0.0;
        while (j < 2.0) {
            var val = (i * 2.0 + j) + 1.0;
            var ii = i as i64;
            var jj = j as i64;
            tensor_store(B, [ii, jj], val);
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    // Compute C = A × B
    var C = tensor_matmul(A, B);

    // Expected C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    var c00 = tensor_load(C, [0i, 0i]);

    return c00;  // Should return 22.0
}

// Test 3: 3D batched matrix multiplication
// A(2x3x4) × B(2x4x3) = C(2x3x3)
fn test_matmul_3d_batched() -> f64 {
    f64<2, 3, 4> A;
    f64<2, 4, 3> B;

    // Initialize A
    var b = 0.0;
    while (b < 2.0) {
        var i = 0.0;
        while (i < 3.0) {
            var j = 0.0;
            while (j < 4.0) {
                var val = (b + 1.0) * (i + j);
                var bb = b as i64;
                var ii = i as i64;
                var jj = j as i64;
                tensor_store(A, [bb, ii, jj], val);
                j = j + 1.0;
            }
            i = i + 1.0;
        }
        b = b + 1.0;
    }

    // Initialize B
    b = 0.0;
    while (b < 2.0) {
        var i = 0.0;
        while (i < 4.0) {
            var j = 0.0;
            while (j < 3.0) {
                var val = (b + 1.0) * (i + 1.0);
                var bb = b as i64;
                var ii = i as i64;
                var jj = j as i64;
                tensor_store(B, [bb, ii, jj], val);
                j = j + 1.0;
            }
            i = i + 1.0;
        }
        b = b + 1.0;
    }

    // Compute C = A × B (batched)
    var C = tensor_matmul(A, B);

    // Return a sample value from batch 0
    var c000 = tensor_load(C, [0i, 0i, 0i]);

    return c000;
}

// Test 4: Small 2D test for easier verification
// A(2x2) × B(2x2) = C(2x2)
fn test_matmul_2x2() -> f64 {
    f64<2, 2> A;
    f64<2, 2> B;

    // Initialize A = [[1, 2], [3, 4]]
    tensor_store(A, [0i, 0i], 1.0);
    tensor_store(A, [0i, 1i], 2.0);
    tensor_store(A, [1i, 0i], 3.0);
    tensor_store(A, [1i, 1i], 4.0);

    // Initialize B = [[1, 0], [0, 1]] (identity)
    tensor_store(B, [0i, 0i], 1.0);
    tensor_store(B, [0i, 1i], 0.0);
    tensor_store(B, [1i, 0i], 0.0);
    tensor_store(B, [1i, 1i], 1.0);

    // Compute C = A × B = A (since B is identity)
    var C = tensor_matmul(A, B);

    // Expected C[0,0] = 1*1 + 2*0 = 1.0
    var c00 = tensor_load(C, [0i, 0i]);

    return c00;  // Should return 1.0
}

// Main test runner
fn kernel_main() -> f64 {
    // Run simplest test first
    var simple_result = test_matmul_2x2();

    return simple_result;  // Should return 1.0
}
