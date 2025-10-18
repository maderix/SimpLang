// SimpLang Tensor Library - Core Operations
// Auto-vectorized tensor operations with proper SimpLang syntax

fn tensor_add_f32(f32, simd=auto[] a, f32, simd=auto[] b, f32, simd=auto[] result, i32 size) -> f32 {
    var i = 0;
    while (i < size) {
        result[i] = a[i] + b[i];
        i = i + 1;
    }
    return 0.0;
}

fn tensor_mul_f32(f32, simd=auto[] a, f32, simd=auto[] b, f32, simd=auto[] result, i32 size) -> f32 {
    var i = 0;
    while (i < size) {
        result[i] = a[i] * b[i];
        i = i + 1;
    }
    return 0.0;
}

fn tensor_sum_f32(f32, simd=auto[] a, i32 size) -> f32 {
    var sum = 0.0;
    var i = 0;
    while (i < size) {
        sum = sum + a[i];
        i = i + 1;
    }
    return sum;
}

fn tensor_fill_f32(f32, simd=auto[] a, f32 value, i32 size) -> f32 {
    var i = 0;
    while (i < size) {
        a[i] = value;
        i = i + 1;
    }
    return 0.0;
}

fn tensor_dot_f32(f32, simd=auto[] a, f32, simd=auto[] b, i32 size) -> f32 {
    var sum = 0.0;
    var i = 0;
    while (i < size) {
        sum = sum + (a[i] * b[i]);
        i = i + 1;
    }
    return sum;
}