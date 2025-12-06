// INT4 GEMM Comprehensive Benchmarks
// Tests i4×i4→i16 matmul across multiple sizes for ML workloads
//
// INT4 provides 2× memory savings vs INT8, ideal for:
// - Weight-only quantization (W4)
// - Extreme compression for edge deployment
// - Cache-efficient inference
//
// Sizes:
// - Small: 32, 64, 128 (embedding layers, attention heads)
// - Medium: 256, 384, 512 (linear layers)
// - Large: 768, 1024 (BERT, transformer FFN)

// ========== Small Sizes ==========

fn benchmark_int4_matmul_32() -> i32 {
    i4<32, 32> A;
    i4<32, 32> B;

    var i = 0;
    while (i < 32) {
        var j = 0;
        while (j < 32) {
            // INT4 range: -8 to 7
            var val = ((i * 32 + j) % 15) - 7;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

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

fn benchmark_int4_matmul_64() -> i32 {
    i4<64, 64> A;
    i4<64, 64> B;

    var i = 0;
    while (i < 64) {
        var j = 0;
        while (j < 64) {
            var val = ((i * 64 + j) % 15) - 7;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
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

fn benchmark_int4_matmul_128() -> i32 {
    i4<128, 128> A;
    i4<128, 128> B;

    var i = 0;
    while (i < 128) {
        var j = 0;
        while (j < 128) {
            var val = ((i * 128 + j) % 15) - 7;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
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

// ========== Medium Sizes ==========

fn benchmark_int4_matmul_256() -> i32 {
    i4<256, 256> A;
    i4<256, 256> B;

    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            var val = ((i * 256 + j) % 15) - 7;
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

fn benchmark_int4_matmul_384() -> i32 {
    i4<384, 384> A;
    i4<384, 384> B;

    var i = 0;
    while (i < 384) {
        var j = 0;
        while (j < 384) {
            var val = ((i * 384 + j) % 15) - 7;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    var C = tensor_matmul(A, B);

    var checksum = 0;
    i = 0;
    while (i < 384) {
        var j = 0;
        while (j < 384) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

fn benchmark_int4_matmul_512() -> i32 {
    i4<512, 512> A;
    i4<512, 512> B;

    var i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            var val = ((i * 512 + j) % 15) - 7;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
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

// ========== Large Sizes ==========

fn benchmark_int4_matmul_768() -> i32 {
    i4<768, 768> A;
    i4<768, 768> B;

    var i = 0;
    while (i < 768) {
        var j = 0;
        while (j < 768) {
            var val = ((i * 768 + j) % 15) - 7;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    var C = tensor_matmul(A, B);

    var checksum = 0;
    i = 0;
    while (i < 768) {
        var j = 0;
        while (j < 768) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

fn benchmark_int4_matmul_1024() -> i32 {
    i4<1024, 1024> A;
    i4<1024, 1024> B;

    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            var val = ((i * 1024 + j) % 15) - 7;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
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

// ========== Entry Point ==========

fn kernel_main() -> i32 {
    return benchmark_int4_matmul_256();
}
