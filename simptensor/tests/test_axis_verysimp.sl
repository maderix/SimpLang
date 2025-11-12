// Very simple test - just call axis reduction, don't access result
fn test_verysimp() -> f32 {
    f32<2,3> t = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    tensor_sum(t, 0);  // Just call it, don't use result
    return 1.0;
}
