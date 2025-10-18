// SimpLang Tensor Library - Core Type Operations
// Proper tensor interface for N-dimensional arrays with auto-vectorization

// ============================================================================
// TENSOR CREATION FUNCTIONS
// ============================================================================

fn tensor_create_f32(i32 size) -> f32, simd=auto[] {
    return array<f32, simd=auto>([size]);
}

fn tensor_create_f64(i32 size) -> f64, simd=auto[] {
    return array<f64, simd=auto>([size]);
}

fn tensor_create_i32(i32 size) -> i32, simd=auto[] {
    return array<i32, simd=auto>([size]);
}

fn tensor_create_i64(i32 size) -> i64, simd=auto[] {
    return array<i64, simd=auto>([size]);
}

// ============================================================================
// TENSOR ELEMENT-WISE OPERATIONS
// ============================================================================

fn tensor_add_f32(f32, simd=auto[] a, f32, simd=auto[] b, f32, simd=auto[] result, i32 size) {
    var i = 0;
    while (i < size) {
        result[i] = a[i] + b[i];
        i = i + 1;
    }
}

fn tensor_mul_f32(f32, simd=auto[] a, f32, simd=auto[] b, f32, simd=auto[] result, i32 size) {
    var i = 0;
    while (i < size) {
        result[i] = a[i] * b[i];
        i = i + 1;
    }
}

fn tensor_sub_f32(f32, simd=auto[] a, f32, simd=auto[] b, f32, simd=auto[] result, i32 size) {
    var i = 0;
    while (i < size) {
        result[i] = a[i] - b[i];
        i = i + 1;
    }
}

fn tensor_div_f32(f32, simd=auto[] a, f32, simd=auto[] b, f32, simd=auto[] result, i32 size) {
    var i = 0;
    while (i < size) {
        result[i] = a[i] / b[i];
        i = i + 1;
    }
}

// ============================================================================
// TENSOR REDUCTION OPERATIONS
// ============================================================================

fn tensor_sum_f32(f32, simd=auto[] a, i32 size) -> f32 {
    var sum = 0.0;
    var i = 0;
    while (i < size) {
        sum = sum + a[i];
        i = i + 1;
    }
    return sum;
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

// ============================================================================
// TENSOR UTILITY OPERATIONS
// ============================================================================

fn tensor_fill_f32(f32, simd=auto[] a, f32 value, i32 size) {
    var i = 0;
    while (i < size) {
        a[i] = value;
        i = i + 1;
    }
}

fn tensor_copy_f32(f32, simd=auto[] src, f32, simd=auto[] dst, i32 size) {
    var i = 0;
    while (i < size) {
        dst[i] = src[i];
        i = i + 1;
    }
}

// ============================================================================
// MATRIX OPERATIONS (2D TENSORS)
// ============================================================================

fn tensor_matmul_f32(f32, simd=auto[] A, f32, simd=auto[] B, f32, simd=auto[] C, 
                     i32 M, i32 K, i32 N) {
    var i = 0;
    while (i < M) {
        var j = 0;
        while (j < N) {
            var sum = 0.0;
            var k = 0;
            while (k < K) {
                var a_idx = i * K + k;
                var b_idx = k * N + j;
                sum = sum + (A[a_idx] * B[b_idx]);
                k = k + 1;
            }
            var c_idx = i * N + j;
            C[c_idx] = sum;
            j = j + 1;
        }
        i = i + 1;
    }
}