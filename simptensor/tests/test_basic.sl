fn test_tensor_basic() -> f32 {
    f32<2,3> t;
    t[0i, 1i] = 5.0;
    var result = t[0i, 1i];
    return result;
}
