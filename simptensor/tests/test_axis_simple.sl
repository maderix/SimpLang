// Simple test for axis reduction
fn test_simple_axis() -> f32 {
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // Sum along axis 0 -> [5, 7, 9]
    f32<2,3> t = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    f32<3> result = tensor_sum(t, 0);
    return result[0i];  // Should be 5.0
}
