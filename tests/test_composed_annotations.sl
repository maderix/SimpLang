// Test file for composed annotations
// Tests individual annotations and composable transform chains
// Run: ./build_mlir/src/simplang tests/test_composed_annotations.sl --emit-mlir --llvm-vectorize -o /tmp/composed.o

// =============================================================================
// Individual Annotation Tests
// =============================================================================

// Test 1: @unroll only
fn test_unroll() -> i32 {
    i32[64] arr;
    var i = 0;
    while (i < 64) {
        arr[i as i64] = i;
        i = i + 1;
    }

    var sum = 0;
    i = 0;
    @unroll(4)
    {
        while (i < 64) {
            sum = sum + arr[i as i64];
            i = i + 1;
        }
    }
    return sum;  // Expected: 0+1+2+...+63 = 2016
}

// Test 2: @parallel only (marks for future OpenMP lowering)
fn test_parallel() -> i32 {
    i8<64, 64> A;
    i8<64, 64> B;
    var i = 0;
    @parallel
    {
        while (i < 64) {
            var j = 0;
            while (j < 64) {
                A[i as i64, j as i64] = 1 as i8;
                B[j as i64, i as i64] = 1 as i8;
                j = j + 1;
            }
            i = i + 1;
        }
    }
    @tile(16, 16, 16) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 64
}

// Test 3: @vectorize width specification
fn test_vectorize() -> i32 {
    i8<64, 64> A;
    i8<64, 64> B;
    var i = 0;
    while (i < 64) {
        var j = 0;
        while (j < 64) {
            A[i as i64, j as i64] = 1 as i8;
            B[j as i64, i as i64] = 1 as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @vectorize(256) @tile(16, 16, 16) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 64
}

// Test 4: @prefetch hint
fn test_prefetch() -> i32 {
    i8<64, 64> A;
    i8<64, 64> B;
    var i = 0;
    while (i < 64) {
        var j = 0;
        while (j < 64) {
            A[i as i64, j as i64] = 1 as i8;
            B[j as i64, i as i64] = 1 as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @prefetch(8) @tile(16, 16, 16) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 64
}

// =============================================================================
// Composed Annotation Tests (multiple annotations chained)
// =============================================================================

// Test 5: @parallel + @tile + @lower (full VNNI pipeline)
fn test_parallel_tile_lower() -> i32 {
    i8<128, 128> A;
    i8<128, 128> B;
    var i = 0;
    while (i < 128) {
        var j = 0;
        while (j < 128) {
            A[i as i64, j as i64] = 1 as i8;
            B[j as i64, i as i64] = 1 as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @parallel @tile(32, 32, 32) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 128
}

// Test 6: @tile + @unroll + @lower (tiled with unrolled inner loops)
fn test_tile_unroll_lower() -> i32 {
    i8<128, 128> A;
    i8<128, 128> B;
    var i = 0;
    while (i < 128) {
        var j = 0;
        while (j < 128) {
            A[i as i64, j as i64] = 1 as i8;
            B[j as i64, i as i64] = 1 as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(32, 32, 32) @unroll(4) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 128
}

// Test 7: Full pipeline - @parallel + @tile + @unroll + @vectorize + @prefetch + @lower
fn test_full_pipeline() -> i32 {
    i8<256, 256> A;
    i8<256, 256> B;
    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            A[i as i64, j as i64] = 1 as i8;
            B[j as i64, i as i64] = 1 as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @parallel @tile(64, 64, 64) @unroll(4) @vectorize(256) @prefetch(8) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 256
}

// Test 8: Different tile sizes with composed annotations
fn test_composed_tiles() -> i32 {
    i8<256, 256> A;
    i8<256, 256> B;
    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            A[i as i64, j as i64] = 1 as i8;
            B[j as i64, i as i64] = 1 as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(16, 16, 16) @unroll(2) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 256
}

// =============================================================================
// Performance benchmark with composed annotations
// =============================================================================

fn benchmark_256_composed() -> i32 {
    i8<256, 256> A;
    i8<256, 256> B;
    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            A[i as i64, j as i64] = ((i + j) % 127) as i8;
            B[j as i64, i as i64] = ((i * j) % 127) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @parallel @tile(16, 16, 16) @unroll(4) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);

    // Compute checksum
    var sum = 0;
    i = 0;
    while (i < 16) {
        var j = 0;
        while (j < 16) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn benchmark_512_composed() -> i32 {
    i8<512, 512> A;
    i8<512, 512> B;
    var i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            A[i as i64, j as i64] = ((i + j) % 127) as i8;
            B[j as i64, i as i64] = ((i * j) % 127) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @parallel @tile(64, 64, 64) @unroll(4) @prefetch(8) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);

    // Compute checksum
    var sum = 0;
    i = 0;
    while (i < 16) {
        var j = 0;
        while (j < 16) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

// Entry point
fn kernel_main() -> i32 {
    return test_parallel_tile_lower();
}
