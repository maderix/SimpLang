// INT8 GEMM Benchmark with VNNI Annotations
// Same as bench_int8_matmul.sl but with @lower("vnni.i8_matmul") annotations

fn benchmark_int8_matmul_32() -> i32 {
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
    @tile(32, 32, 32) @lower("vnni.i8_matmul")
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

fn benchmark_int8_matmul_64() -> i32 {
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
    @tile(32, 32, 32) @lower("vnni.i8_matmul")
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

fn benchmark_int8_matmul_128() -> i32 {
    i8<128, 128> A;
    i8<128, 128> B;
    var i = 0;
    while (i < 128) {
        var j = 0;
        while (j < 128) {
            var val = ((i * 128 + j) % 127) - 64;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
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

fn benchmark_int8_matmul_256() -> i32 {
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
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
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

fn benchmark_int8_matmul_384() -> i32 {
    i8<384, 384> A;
    i8<384, 384> B;
    var i = 0;
    while (i < 384) {
        var j = 0;
        while (j < 384) {
            var val = ((i * 384 + j) % 127) - 64;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
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

fn benchmark_int8_matmul_512() -> i32 {
    i8<512, 512> A;
    i8<512, 512> B;
    var i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            var val = ((i * 512 + j) % 127) - 64;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
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

fn benchmark_int8_matmul_768() -> i32 {
    i8<768, 768> A;
    i8<768, 768> B;
    var i = 0;
    while (i < 768) {
        var j = 0;
        while (j < 768) {
            var val = ((i * 768 + j) % 127) - 64;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
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

fn benchmark_int8_matmul_1024() -> i32 {
    i8<1024, 1024> A;
    i8<1024, 1024> B;
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            var val = ((i * 1024 + j) % 127) - 64;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
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

fn benchmark_int8_matmul_2048() -> i32 {
    i8<2048, 2048> A;
    i8<2048, 2048> B;
    var i = 0;
    while (i < 2048) {
        var j = 0;
        while (j < 2048) {
            var val = ((i * 2048 + j) % 127) - 64;
            A[i as i64, j as i64] = val;
            B[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var checksum = 0;
    i = 0;
    while (i < 2048) {
        var j = 0;
        while (j < 2048) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return checksum;
}

fn kernel_main() -> i32 {
    return benchmark_int8_matmul_256();
}
