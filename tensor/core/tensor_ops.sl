// SimpLang Tensor Library - Extended Operations
// Additional tensor operations building on core types

// ============================================================================
// INTEGER TENSOR OPERATIONS
// ============================================================================

fn tensor_add_i32(i32, simd=auto[] a, i32, simd=auto[] b, i32, simd=auto[] result, i32 size) {
    var i = 0;
    while (i < size) {
        result[i] = a[i] + b[i];
        i = i + 1;
    }
}

fn tensor_mul_i32(i32, simd=auto[] a, i32, simd=auto[] b, i32, simd=auto[] result, i32 size) {
    var i = 0;
    while (i < size) {
        result[i] = a[i] * b[i];
        i = i + 1;
    }
}

fn tensor_sum_i32(i32, simd=auto[] a, i32 size) -> i32 {
    var sum = 0;
    var i = 0;
    while (i < size) {
        sum = sum + a[i];
        i = i + 1;
    }
    return sum;
}

// ============================================================================
// DOUBLE PRECISION OPERATIONS
// ============================================================================

fn tensor_add_f64(f64, simd=auto[] a, f64, simd=auto[] b, f64, simd=auto[] result, i32 size) {
    var i = 0;
    while (i < size) {
        result[i] = a[i] + b[i];
        i = i + 1;
    }
}

fn tensor_mul_f64(f64, simd=auto[] a, f64, simd=auto[] b, f64, simd=auto[] result, i32 size) {
    var i = 0;
    while (i < size) {
        result[i] = a[i] * b[i];
        i = i + 1;
    }
}

fn tensor_sum_f64(f64, simd=auto[] a, i32 size) -> f64 {
    var sum = 0.0;
    var i = 0;
    while (i < size) {
        sum = sum + a[i];
        i = i + 1;
    }
    return sum;
}

// ============================================================================
// SCALAR OPERATIONS
// ============================================================================

fn tensor_add_scalar_f32(f32, simd=auto[] a, f32 scalar, f32, simd=auto[] result, i32 size) {
    var i = 0;
    while (i < size) {
        result[i] = a[i] + scalar;
        i = i + 1;
    }
}

fn tensor_mul_scalar_f32(f32, simd=auto[] a, f32 scalar, f32, simd=auto[] result, i32 size) {
    var i = 0;
    while (i < size) {
        result[i] = a[i] * scalar;
        i = i + 1;
    }
}

// ============================================================================
// COMPARISON OPERATIONS
// ============================================================================

fn tensor_max_f32(f32, simd=auto[] a, i32 size) -> f32 {
    if (size == 0) {
        return 0.0;
    }
    var max_val = a[0];
    var i = 1;
    while (i < size) {
        if (a[i] > max_val) {
            max_val = a[i];
        }
        i = i + 1;
    }
    return max_val;
}

fn tensor_min_f32(f32, simd=auto[] a, i32 size) -> f32 {
    if (size == 0) {
        return 0.0;
    }
    var min_val = a[0];
    var i = 1;
    while (i < size) {
        if (a[i] < min_val) {
            min_val = a[i];
        }
        i = i + 1;
    }
    return min_val;
}

// ============================================================================
// ADVANCED OPERATIONS
// ============================================================================

fn tensor_mean_f32(f32, simd=auto[] a, i32 size) -> f32 {
    var sum = tensor_sum_f32(a, size);
    return sum / size;
}

fn tensor_transpose_f32(f32, simd=auto[] A, f32, simd=auto[] B, i32 M, i32 N) {
    var i = 0;
    while (i < M) {
        var j = 0;
        while (j < N) {
            var a_idx = i * N + j;
            var b_idx = j * M + i;
            B[b_idx] = A[a_idx];
            j = j + 1;
        }
        i = i + 1;
    }
}