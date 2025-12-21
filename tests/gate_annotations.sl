// Gating tests for annotation framework
// Tests accuracy (checksums) and performance baselines
// Run with: ./build_mlir/src/simplang tests/gate_annotations.sl --emit-mlir --llvm-vectorize -o /tmp/gate.o

// =============================================================================
// ACCURACY GATES - Must produce correct checksums
// =============================================================================

// Gate 1: Basic tiled matmul - 64x64 with 8x8x8 tiles
fn gate_accuracy_tile8() -> i32 {
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
    @tile(8, 8, 8) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 64
}

// Gate 2: Larger tiles - 64x64 with 32x32x32 tiles
fn gate_accuracy_tile32() -> i32 {
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
    @tile(32, 32, 32) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 64
}

// Gate 3: 256x256 matrix for performance baseline
fn gate_accuracy_256() -> i32 {
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
    @tile(16, 16, 16) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 256
}

// Gate 4: 512x512 matrix - larger scale
fn gate_accuracy_512() -> i32 {
    i8<512, 512> A;
    i8<512, 512> B;
    var i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            A[i as i64, j as i64] = 1 as i8;
            B[j as i64, i as i64] = 1 as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    return C[0 as i64, 0 as i64];  // Expected: 512
}

// =============================================================================
// PERFORMANCE GATES - Track GIOP/s baselines
// =============================================================================

// Perf Gate 1: 256x256 @ tile=16 (known good config)
fn gate_perf_256_t16() -> i32 {
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
    @tile(16, 16, 16) @lower("vnni.i8_matmul")
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

// Perf Gate 2: 512x512 @ tile=64 (known good config)
fn gate_perf_512_t64() -> i32 {
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
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
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

// Perf Gate 3: 1024x1024 @ tile=128 (large scale)
fn gate_perf_1024_t128() -> i32 {
    i8<1024, 1024> A;
    i8<1024, 1024> B;
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            A[i as i64, j as i64] = ((i + j) % 127) as i8;
            B[j as i64, i as i64] = ((i * j) % 127) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(128, 128, 128) @lower("vnni.i8_matmul")
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

// Entry point - just run tile8 test (runner tests all functions)
fn kernel_main() -> i32 {
    return gate_accuracy_tile8();
}
