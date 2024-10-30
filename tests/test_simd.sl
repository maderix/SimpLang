fn test_sse_add() {
    var a = sse(1.0, 2.0, 3.0, 4.0);
    var b = sse(5.0, 6.0, 7.0, 8.0);
    return simd_add(a, b);
}

fn test_sse_mul() {
    var a = sse(1.0, 2.0, 3.0, 4.0);
    var b = sse(2.0, 3.0, 4.0, 5.0);
    return simd_mul(a, b);
}

fn test_avx_add() {
    var a = avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    var b = avx(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    return simd_add(a, b);
}

fn test_avx_mul() {
    var a = avx(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    var b = avx(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    return simd_mul(a, b);
}

fn main() {
    var res1 = test_sse_add();
    var res2 = test_sse_mul();
    var res3 = test_avx_add();
    var res4 = test_avx_mul();
    return 0;
}