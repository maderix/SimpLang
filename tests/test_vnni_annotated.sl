// Test annotation-driven VNNI lowering for INT8 matmul
// This tests the full pipeline: @lower("vnni.i8_matmul") -> tiled matmul -> vpdpbusd

// Test with 32x32 matrices and VNNI annotation
fn test_vnni_32() -> i32 {
    i8<32, 32> A;
    i8<32, 32> B;

    // Initialize matrices
    var i = 0;
    while (i < 32) {
        var j = 0;
        while (j < 32) {
            var val = ((i * 32 + j) % 127) - 64;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    // Use VNNI annotation for optimized matmul
    @tile(32, 32, 32)
    @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);

    // Compute checksum
    var checksum = 0;
    i = 0;
    while (i < 32) {
        var j = 0;
        while (j < 32) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

// Test with 64x64 matrices
fn test_vnni_64() -> i32 {
    i8<64, 64> A;
    i8<64, 64> B;

    var i = 0;
    while (i < 64) {
        var j = 0;
        while (j < 64) {
            var val = ((i * 64 + j) % 127) - 64;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    @tile(32, 32, 32)
    @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);

    var checksum = 0;
    i = 0;
    while (i < 64) {
        var j = 0;
        while (j < 64) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

// Test with 256x256 matrices (larger tile)
fn test_vnni_256() -> i32 {
    i8<256, 256> A;
    i8<256, 256> B;

    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            var val = ((i * 256 + j) % 127) - 64;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    @tile(64, 64, 64)
    @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);

    var checksum = 0;
    i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

// Scalar baseline (no VNNI optimization)
fn test_scalar_baseline() -> i32 {
    i8<32, 32> A;
    i8<32, 32> B;

    var i = 0;
    while (i < 32) {
        var j = 0;
        while (j < 32) {
            var val = ((i * 32 + j) % 127) - 64;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    // Scalar baseline - no annotation
    var C = tensor_matmul(A, B);

    var checksum = 0;
    i = 0;
    while (i < 32) {
        var j = 0;
        while (j < 32) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

fn kernel_main() -> i32 {
    return test_vnni_32();
}
