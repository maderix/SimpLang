// Comprehensive Array Access Pattern Benchmark
// Tests: sequential, random, strided, gather, scatter, block copy, transpose, reduction
// Datatypes: i8, i16, i32, i64, f16, bf16, f32, f64
// Array sizes: 8K (L1), 64K (L2), 1M (L3), 10M (main memory)

// ========== F64 BENCHMARKS ==========

fn bench_f64_sequential_read_10m() -> f64 {
    var N = 10000000i;
    var arr = array<f64>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = 0.0;
    i = 0i;
    while (i < N) { sum = sum + arr[i]; i = i + 1i; }
    return sum;
}

fn bench_f64_sequential_write_10m() -> f64 {
    var N = 10000000i;
    var arr = array<f64>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i * 2i; i = i + 1i; }
    return arr[N - 1i];
}

fn bench_f64_strided_read_stride4() -> f64 {
    var N = 1048576i;
    var arr = array<f64>([1048576]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = 0.0;
    i = 0i;
    while (i < N) { sum = sum + arr[i]; i = i + 4i; }
    return sum;
}

fn bench_f64_random_read() -> f64 {
    var N = 65536i;
    var arr = array<f64>([65536]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = 0.0;
    var seed = 12345i;
    i = 0i;
    while (i < N) {
        var temp = 1103515245i * seed + 12345i;
        seed = temp - (temp / 65536i) * 65536i;
        var idx = seed - (seed / N) * N;
        sum = sum + arr[idx];
        i = i + 1i;
    }
    return sum;
}

fn bench_f64_gather() -> f64 {
    var N = 65536i;
    var src = array<f64>([65536]);
    var indices = array<i64>([65536]);
    var dst = array<f64>([65536]);
    var i = 0i;
    while (i < N) { src[i] = i * 2i; indices[i] = N - 1i - i; i = i + 1i; }
    i = 0i;
    while (i < N) { dst[i] = src[indices[i]]; i = i + 1i; }
    return dst[0i];
}

fn bench_f64_scatter() -> f64 {
    var N = 65536i;
    var src = array<f64>([65536]);
    var indices = array<i64>([65536]);
    var dst = array<f64>([65536]);
    var i = 0i;
    while (i < N) { src[i] = i * 3i; indices[i] = N - 1i - i; dst[i] = 0.0; i = i + 1i; }
    i = 0i;
    while (i < N) { dst[indices[i]] = src[i]; i = i + 1i; }
    return dst[N - 1i];
}

fn bench_f64_block_copy_10m() -> f64 {
    var N = 10000000i;
    var src = array<f64>([10000000]);
    var dst = array<f64>([10000000]);
    var i = 0i;
    while (i < N) { src[i] = i; i = i + 1i; }
    i = 0i;
    while (i < N) { dst[i] = src[i]; i = i + 1i; }
    return dst[N - 1i];
}

fn bench_f64_transpose() -> f64 {
    var rows = 512i;
    var cols = 512i;
    var src = array<f64>([512, 512]);
    var dst = array<f64>([512, 512]);
    var i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) { src[i, j] = i * cols + j; j = j + 1i; }
        i = i + 1i;
    }
    i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) { dst[j, i] = src[i, j]; j = j + 1i; }
        i = i + 1i;
    }
    return dst[0i, rows - 1i];
}

// ========== F32 BENCHMARKS ==========

fn bench_f32_sequential_read_10m() -> f32 {
    var N = 10000000i;
    var arr = array<f32>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = arr[0i];
    i = 0i;
    while (i < N) { sum = sum + arr[i]; i = i + 1i; }
    return sum;
}

fn bench_f32_sequential_write_10m() -> f32 {
    var N = 10000000i;
    var arr = array<f32>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i * 2i; i = i + 1i; }
    return arr[N - 1i];
}

fn bench_f32_strided_read_stride4() -> f32 {
    var N = 1048576i;
    var arr = array<f32>([1048576]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = arr[0i];
    i = 0i;
    while (i < N) { sum = sum + arr[i]; i = i + 4i; }
    return sum;
}

fn bench_f32_block_copy_10m() -> f32 {
    var N = 10000000i;
    var src = array<f32>([10000000]);
    var dst = array<f32>([10000000]);
    var i = 0i;
    while (i < N) { src[i] = i; i = i + 1i; }
    i = 0i;
    while (i < N) { dst[i] = src[i]; i = i + 1i; }
    return dst[N - 1i];
}

// ========== I64 BENCHMARKS ==========

fn bench_i64_sequential_read_10m() -> i64 {
    var N = 10000000i;
    var arr = array<i64>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = 0i;
    i = 0i;
    while (i < N) { sum = sum + arr[i]; i = i + 1i; }
    return sum;
}

fn bench_i64_sequential_write_10m() -> i64 {
    var N = 10000000i;
    var arr = array<i64>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i * 2i; i = i + 1i; }
    return arr[N - 1i];
}

fn bench_i64_strided_read_stride4() -> i64 {
    var N = 1048576i;
    var arr = array<i64>([1048576]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = 0i;
    i = 0i;
    while (i < N) { sum = sum + arr[i]; i = i + 4i; }
    return sum;
}

fn bench_i64_block_copy_10m() -> i64 {
    var N = 10000000i;
    var src = array<i64>([10000000]);
    var dst = array<i64>([10000000]);
    var i = 0i;
    while (i < N) { src[i] = i; i = i + 1i; }
    i = 0i;
    while (i < N) { dst[i] = src[i]; i = i + 1i; }
    return dst[N - 1i];
}

// ========== I32 BENCHMARKS ==========

fn bench_i32_sequential_read_10m() -> i32 {
    var N = 10000000i;
    var arr = array<i32>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = 0;
    i = 0i;
    while (i < N) { sum = sum + arr[i]; i = i + 1i; }
    return sum;
}

fn bench_i32_sequential_write_10m() -> i32 {
    var N = 10000000i;
    var arr = array<i32>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i * 2i; i = i + 1i; }
    return arr[N - 1i];
}

fn bench_i32_block_copy_10m() -> i32 {
    var N = 10000000i;
    var src = array<i32>([10000000]);
    var dst = array<i32>([10000000]);
    var i = 0i;
    while (i < N) { src[i] = i; i = i + 1i; }
    i = 0i;
    while (i < N) { dst[i] = src[i]; i = i + 1i; }
    return dst[N - 1i];
}

// ========== I16 BENCHMARKS ==========

fn bench_i16_sequential_read_10m() -> i16 {
    var N = 10000000i;
    var arr = array<i16>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = arr[0i];
    i = 0i;
    while (i < N) { sum = sum + arr[i]; i = i + 1i; }
    return sum;
}

fn bench_i16_sequential_write_10m() -> i16 {
    var N = 10000000i;
    var arr = array<i16>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    return arr[N - 1i];
}

fn bench_i16_block_copy_10m() -> i16 {
    var N = 10000000i;
    var src = array<i16>([10000000]);
    var dst = array<i16>([10000000]);
    var i = 0i;
    while (i < N) { src[i] = i; i = i + 1i; }
    i = 0i;
    while (i < N) { dst[i] = src[i]; i = i + 1i; }
    return dst[N - 1i];
}

// ========== I8 BENCHMARKS ==========

fn bench_i8_sequential_read_10m() -> i8 {
    var N = 10000000i;
    var arr = array<i8>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = arr[0i];
    i = 0i;
    while (i < N) { sum = sum + arr[i]; i = i + 1i; }
    return sum;
}

fn bench_i8_sequential_write_10m() -> i8 {
    var N = 10000000i;
    var arr = array<i8>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    return arr[N - 1i];
}

fn bench_i8_block_copy_10m() -> i8 {
    var N = 10000000i;
    var src = array<i8>([10000000]);
    var dst = array<i8>([10000000]);
    var i = 0i;
    while (i < N) { src[i] = i; i = i + 1i; }
    i = 0i;
    while (i < N) { dst[i] = src[i]; i = i + 1i; }
    return dst[N - 1i];
}

// ========== F16 BENCHMARKS ==========

fn bench_f16_sequential_read_10m() -> f16 {
    var N = 10000000i;
    var arr = array<f16>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    var sum = arr[0i];
    i = 0i;
    while (i < N) { sum = sum + arr[i]; i = i + 1i; }
    return sum;
}

fn bench_f16_sequential_write_10m() -> f16 {
    var N = 10000000i;
    var arr = array<f16>([10000000]);
    var i = 0i;
    while (i < N) { arr[i] = i; i = i + 1i; }
    return arr[N - 1i];
}

fn bench_f16_block_copy_10m() -> f16 {
    var N = 10000000i;
    var src = array<f16>([10000000]);
    var dst = array<f16>([10000000]);
    var i = 0i;
    while (i < N) { src[i] = i; i = i + 1i; }
    i = 0i;
    while (i < N) { dst[i] = src[i]; i = i + 1i; }
    return dst[N - 1i];
}

// ========== BF16 BENCHMARKS ==========
// SKIPPED: BF16 has LLVM backend issues with i64->bf16 conversion

fn kernel_main() -> f64 {
    return bench_f64_sequential_read_10m();
}
