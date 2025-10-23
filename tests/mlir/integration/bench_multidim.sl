// Benchmark: Multi-dimensional array indexing performance
// Compares 1D flattened access vs 2D/3D indexed access

// Baseline: 1D flattened access (direct indexing)
fn bench_1d_baseline() -> f64 {
    var N = 1000i;
    var arr = array<f64>([1000]);
    
    // Write loop
    var i = 0i;
    while (i < N) {
        arr[i] = i;
        i = i + 1i;
    }
    
    // Read and sum
    var sum = 0.0;
    i = 0i;
    while (i < N) {
        sum = sum + arr[i];
        i = i + 1i;
    }
    
    return sum;  // Expected: 499500.0
}

// 2D array with multi-dimensional indexing (32x32 = 1024 elements)
fn bench_2d_indexed() -> f64 {
    var rows = 32i;
    var cols = 32i;
    var arr = array<f64>([32, 32]);
    
    // Write loop with 2D indexing
    var i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            var val = i * cols + j;  // Compute value
            arr[i, j] = val;
            j = j + 1i;
        }
        i = i + 1i;
    }
    
    // Read and sum with 2D indexing
    var sum = 0.0;
    i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            sum = sum + arr[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }
    
    return sum;  // Expected: 523776.0 (sum of 0..1023)
}

// 2D array with manual flattening (baseline for comparison)
fn bench_2d_flattened() -> f64 {
    var rows = 32i;
    var cols = 32i;
    var total = 1024i;
    var arr = array<f64>([1024]);
    
    // Write loop with manual index computation
    var i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            var idx = i * cols + j;
            var val = i * cols + j;
            arr[idx] = val;
            j = j + 1i;
        }
        i = i + 1i;
    }
    
    // Read and sum with manual flattening
    var sum = 0.0;
    i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            var idx = i * cols + j;
            sum = sum + arr[idx];
            j = j + 1i;
        }
        i = i + 1i;
    }
    
    return sum;  // Expected: 523776.0
}

// 3D array with multi-dimensional indexing (16x16x4 = 1024 elements)
fn bench_3d_indexed() -> f64 {
    var d0 = 16i;
    var d1 = 16i;
    var d2 = 4i;
    var arr = array<f64>([16, 16, 4]);
    
    // Write loop
    var i = 0i;
    while (i < d0) {
        var j = 0i;
        while (j < d1) {
            var k = 0i;
            while (k < d2) {
                var val = (i * d1 * d2) + (j * d2) + k;
                arr[i, j, k] = val;
                k = k + 1i;
            }
            j = j + 1i;
        }
        i = i + 1i;
    }
    
    // Read and sum
    var sum = 0.0;
    i = 0i;
    while (i < d0) {
        var j = 0i;
        while (j < d1) {
            var k = 0i;
            while (k < d2) {
                sum = sum + arr[i, j, k];
                k = k + 1i;
            }
            j = j + 1i;
        }
        i = i + 1i;
    }
    
    return sum;  // Expected: 523776.0
}

// INTEGER BENCHMARKS: Test integer type promotion (i32 -> i64)

// 2D integer array with multi-dimensional indexing (256x256 = 65536 elements)
fn bench_2d_int_indexed() -> i64 {
    var rows = 256i;
    var cols = 256i;
    var arr = array<i64>([256, 256]);

    // Write loop with 2D indexing
    var i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            var val = i * cols + j;
            arr[i, j] = val;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Read and sum with 2D indexing - i32 accumulator + i64 array
    var sum = 0i;  // i32 will be promoted to i64
    i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            sum = sum + arr[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }

    return sum;  // Expected: 2147450880 (sum of 0..65535)
}

// 2D integer array with manual flattening (baseline)
fn bench_2d_int_flattened() -> i64 {
    var rows = 256i;
    var cols = 256i;
    var total = 65536i;
    var arr = array<i64>([65536]);

    // Write loop with manual index computation
    var i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            var idx = i * cols + j;
            var val = i * cols + j;
            arr[idx] = val;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Read and sum with manual flattening - i32 accumulator + i64 array
    var sum = 0i;  // i32 will be promoted to i64
    i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            var idx = i * cols + j;
            sum = sum + arr[idx];
            j = j + 1i;
        }
        i = i + 1i;
    }

    return sum;  // Expected: 2147450880
}

// LARGE ARRAY BENCHMARKS: Stress cache hierarchy (L1/L2/L3)

// Large 2D float array with multi-dimensional indexing (512x512 = 262144 elements)
fn bench_2d_large_indexed() -> f64 {
    var rows = 512i;
    var cols = 512i;
    var arr = array<f64>([512, 512]);

    // Write loop with 2D indexing
    var i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            var val = i * cols + j;
            arr[i, j] = val;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Read and sum with 2D indexing
    var sum = 0.0;
    i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            sum = sum + arr[i, j];
            j = j + 1i;
        }
        i = i + 1i;
    }

    return sum;  // Expected: 34359607296.0 (sum of 0..262143)
}

// Large 2D float array with manual flattening (baseline)
fn bench_2d_large_flattened() -> f64 {
    var rows = 512i;
    var cols = 512i;
    var total = 262144i;
    var arr = array<f64>([262144]);

    // Write loop with manual index computation
    var i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            var idx = i * cols + j;
            var val = i * cols + j;
            arr[idx] = val;
            j = j + 1i;
        }
        i = i + 1i;
    }

    // Read and sum with manual flattening
    var sum = 0.0;
    i = 0i;
    while (i < rows) {
        var j = 0i;
        while (j < cols) {
            var idx = i * cols + j;
            sum = sum + arr[idx];
            j = j + 1i;
        }
        i = i + 1i;
    }

    return sum;  // Expected: 34359607296.0
}

fn kernel_main() -> f64 {
    // Run all benchmarks and return sum for verification
    var r1 = bench_1d_baseline();
    var r2 = bench_2d_indexed();
    var r3 = bench_2d_flattened();
    var r4 = bench_3d_indexed();

    return r1 + r2 + r3 + r4;
}
