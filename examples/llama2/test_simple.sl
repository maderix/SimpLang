// Minimal test to check parameter passing

fn test_params(
    f32[] a,
    f32[] b,
    f32[] c,
    i64 p1, i64 p2, i64 p3, i64 p4, i64 p5,
    i64 p6, i64 p7, i64 p8, i64 p9, i64 p10
) -> f32 {
    var x = p1 / p2;
    var y = p3 / p4;
    return a[0i] + b[0i] + c[0i];
}
