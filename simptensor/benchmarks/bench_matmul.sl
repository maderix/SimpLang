// GEMM benchmarks for representative ML workload sizes
// Sizes: 64 (tiny), 128 (small), 256 (medium), 512 (large), 1024 (very large)

fn benchmark_matmul_64() -> f32 {
    f32<64, 64> A;
    f32<64, 64> B;

    var i = 0.0;
    while (i < 64.0) {
        var j = 0.0;
        while (j < 64.0) {
            A[i as i64, j as i64] = (i * 64.0 + j) / 4096.0;
            B[i as i64, j as i64] = (j * 64.0 + i) / 4096.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    var C = tensor_matmul(A, B);

    var checksum = 0.0;
    i = 0.0;
    while (i < 64.0) {
        var j = 0.0;
        while (j < 64.0) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    return checksum;
}

fn benchmark_matmul_128() -> f32 {
    f32<128, 128> A;
    f32<128, 128> B;

    var i = 0.0;
    while (i < 128.0) {
        var j = 0.0;
        while (j < 128.0) {
            A[i as i64, j as i64] = (i * 128.0 + j) / 16384.0;
            B[i as i64, j as i64] = (j * 128.0 + i) / 16384.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    var C = tensor_matmul(A, B);

    var checksum = 0.0;
    i = 0.0;
    while (i < 128.0) {
        var j = 0.0;
        while (j < 128.0) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    return checksum;
}

fn benchmark_matmul_256() -> f32 {
    // Initialize 256x256 matrices with sequential values
    f32<256, 256> A;
    f32<256, 256> B;

    var i = 0.0;
    while (i < 256.0) {
        var j = 0.0;
        while (j < 256.0) {
            var idx_i = i as i64;
            var idx_j = j as i64;
            A[idx_i, idx_j] = (i * 256.0 + j) / 65536.0;  // Normalize to [0, 1]
            B[idx_i, idx_j] = (j * 256.0 + i) / 65536.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    // Perform matrix multiplication
    var C = tensor_matmul(A, B);

    // Return checksum to verify correctness
    var checksum = 0.0;
    i = 0.0;
    while (i < 256.0) {
        var j = 0.0;
        while (j < 256.0) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    return checksum;
}

fn benchmark_matmul_512() -> f32 {
    f32<512, 512> A;
    f32<512, 512> B;

    var i = 0.0;
    while (i < 512.0) {
        var j = 0.0;
        while (j < 512.0) {
            A[i as i64, j as i64] = (i * 512.0 + j) / 262144.0;
            B[i as i64, j as i64] = (j * 512.0 + i) / 262144.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    var C = tensor_matmul(A, B);

    var checksum = 0.0;
    i = 0.0;
    while (i < 512.0) {
        var j = 0.0;
        while (j < 512.0) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    return checksum;
}

fn benchmark_matmul_1024() -> f32 {
    f32<1024, 1024> A;
    f32<1024, 1024> B;

    var i = 0.0;
    while (i < 1024.0) {
        var j = 0.0;
        while (j < 1024.0) {
            A[i as i64, j as i64] = (i * 1024.0 + j) / 1048576.0;
            B[i as i64, j as i64] = (j * 1024.0 + i) / 1048576.0;
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    var C = tensor_matmul(A, B);

    var checksum = 0.0;
    i = 0.0;
    while (i < 1024.0) {
        var j = 0.0;
        while (j < 1024.0) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1.0;
        }
        i = i + 1.0;
    }

    return checksum;
}

// Integer matmul benchmarks

fn benchmark_matmul_256_i8() -> i32 {
    i8<256, 256> A;
    i8<256, 256> B;

    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            var val = (i * 256 + j) / 256;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

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

fn benchmark_matmul_256_i16() -> i32 {
    i16<256, 256> A;
    i16<256, 256> B;

    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            var val = (i * 256 + j) / 256;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

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

fn benchmark_matmul_256_i32() -> i32 {
    i32<256, 256> A;
    i32<256, 256> B;

    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            A[i as i64, j as i64] = (i * 256 + j) / 256;
            B[i as i64, j as i64] = (j * 256 + i) / 256;
            j = j + 1;
        }
        i = i + 1;
    }

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

fn benchmark_matmul_64_i32() -> i32 {
    i32<64, 64> A;
    i32<64, 64> B;

    var i = 0;
    while (i < 64) {
        var j = 0;
        while (j < 64) {
            A[i as i64, j as i64] = (i * 64 + j) / 64;
            B[i as i64, j as i64] = (j * 64 + i) / 64;
            j = j + 1;
        }
        i = i + 1;
    }

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

fn benchmark_matmul_128_i32() -> i32 {
    i32<128, 128> A;
    i32<128, 128> B;

    var i = 0;
    while (i < 128) {
        var j = 0;
        while (j < 128) {
            A[i as i64, j as i64] = (i * 128 + j) / 128;
            B[i as i64, j as i64] = (j * 128 + i) / 128;
            j = j + 1;
        }
        i = i + 1;
    }

    var C = tensor_matmul(A, B);

    var checksum = 0;
    i = 0;
    while (i < 128) {
        var j = 0;
        while (j < 128) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

fn benchmark_matmul_512_i32() -> i32 {
    i32<512, 512> A;
    i32<512, 512> B;

    var i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            A[i as i64, j as i64] = (i * 512 + j) / 512;
            B[i as i64, j as i64] = (j * 512 + i) / 512;
            j = j + 1;
        }
        i = i + 1;
    }

    var C = tensor_matmul(A, B);

    var checksum = 0;
    i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

fn benchmark_matmul_1024_i32() -> i32 {
    i32<1024, 1024> A;
    i32<1024, 1024> B;

    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            A[i as i64, j as i64] = (i * 1024 + j) / 1024;
            B[i as i64, j as i64] = (j * 1024 + i) / 1024;
            j = j + 1;
        }
        i = i + 1;
    }

    var C = tensor_matmul(A, B);

    var checksum = 0;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

fn benchmark_matmul_256_i64() -> i64 {
    i64<256, 256> A;
    i64<256, 256> B;

    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            A[i, j] = (i * 256 + j) / 256;
            B[i, j] = (j * 256 + i) / 256;
            j = j + 1;
        }
        i = i + 1;
    }

    var C = tensor_matmul(A, B);

    var checksum = 0;
    i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            checksum = checksum + C[i, j];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

fn kernel_main() -> f32 {
    return benchmark_matmul_256();
}
