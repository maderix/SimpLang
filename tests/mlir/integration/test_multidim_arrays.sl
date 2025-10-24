// Test multi-dimensional array creation and indexing
// Tests 2D, 3D, and 4D arrays with flattened index computation

fn test_2d_array() -> f64 {
    // Create 2D array: 3x4 = 12 elements
    var arr2d = array<f64>([3, 4]);

    // Set element at [1, 2] (flattened index: 1*4 + 2 = 6)
    arr2d[1i, 2i] = 42.0;

    // Read it back
    return arr2d[1i, 2i];  // Should return 42.0
}

fn test_3d_array() -> f64 {
    // Create 3D array: 2x3x4 = 24 elements
    var arr3d = array<f64>([2, 3, 4]);

    // Set element at [1, 2, 3] (flattened: 1*(3*4) + 2*4 + 3 = 12 + 8 + 3 = 23)
    arr3d[1i, 2i, 3i] = 123.0;

    // Read it back
    return arr3d[1i, 2i, 3i];  // Should return 123.0
}

fn test_4d_array() -> f64 {
    // Create 4D array: 2x2x2x2 = 16 elements (small for testing)
    var arr4d = array<f64>([2, 2, 2, 2]);

    // Set element at [1, 0, 1, 1] (flattened: 1*8 + 0*4 + 1*2 + 1 = 8 + 0 + 2 + 1 = 11)
    arr4d[1i, 0i, 1i, 1i] = 999.0;

    // Read it back
    return arr4d[1i, 0i, 1i, 1i];  // Should return 999.0
}

fn test_multidim_iteration() -> f64 {
    // Create 2x3 array
    var arr = array<f64>([2, 3]);

    // Fill with loop (flattened)
    var i = 0i;
    while (i < 6i) {
        arr[i] = i;  // 1D access for filling
        i = i + 1i;
    }

    // Read using 2D indices
    // arr[0,0]=0, arr[0,1]=1, arr[0,2]=2
    // arr[1,0]=3, arr[1,1]=4, arr[1,2]=5
    var sum = arr[0i, 0i] + arr[0i, 1i] + arr[0i, 2i] +
              arr[1i, 0i] + arr[1i, 1i] + arr[1i, 2i];

    return sum;  // Should return 0+1+2+3+4+5 = 15.0
}

// Main entry point for testing
fn kernel_main() -> f64 {
    var result2d = test_2d_array();
    var result3d = test_3d_array();
    var result4d = test_4d_array();
    var result_iter = test_multidim_iteration();

    // Return sum of all results for verification
    // Expected: 42 + 123 + 999 + 15 = 1179.0
    return result2d + result3d + result4d + result_iter;
}
