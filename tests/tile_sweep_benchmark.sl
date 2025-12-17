// Tile Size Sweep Benchmark
// Tests different tile sizes on various matrix dimensions
// to find optimal tiling for different workloads

// ============================================================
// 256×256 Matrix - Tile Size Sweep
// ============================================================

fn matmul_256_tile8() -> i32 {
    i8<256, 256> A;
    i8<256, 256> B;
    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            A[i as i64, j as i64] = ((i * 256 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 256 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(8, 8, 8) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_256_tile16() -> i32 {
    i8<256, 256> A;
    i8<256, 256> B;
    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            A[i as i64, j as i64] = ((i * 256 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 256 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(16, 16, 16) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_256_tile32() -> i32 {
    i8<256, 256> A;
    i8<256, 256> B;
    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            A[i as i64, j as i64] = ((i * 256 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 256 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(32, 32, 32) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_256_tile64() -> i32 {
    i8<256, 256> A;
    i8<256, 256> B;
    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            A[i as i64, j as i64] = ((i * 256 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 256 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_256_tile128() -> i32 {
    i8<256, 256> A;
    i8<256, 256> B;
    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            A[i as i64, j as i64] = ((i * 256 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 256 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(128, 128, 128) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

// ============================================================
// 512×512 Matrix - Tile Size Sweep
// ============================================================

fn matmul_512_tile16() -> i32 {
    i8<512, 512> A;
    i8<512, 512> B;
    var i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            A[i as i64, j as i64] = ((i * 512 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 512 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(16, 16, 16) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_512_tile32() -> i32 {
    i8<512, 512> A;
    i8<512, 512> B;
    var i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            A[i as i64, j as i64] = ((i * 512 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 512 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(32, 32, 32) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_512_tile64() -> i32 {
    i8<512, 512> A;
    i8<512, 512> B;
    var i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            A[i as i64, j as i64] = ((i * 512 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 512 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_512_tile128() -> i32 {
    i8<512, 512> A;
    i8<512, 512> B;
    var i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            A[i as i64, j as i64] = ((i * 512 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 512 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(128, 128, 128) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_512_tile256() -> i32 {
    i8<512, 512> A;
    i8<512, 512> B;
    var i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            A[i as i64, j as i64] = ((i * 512 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 512 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(256, 256, 256) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 512) {
        var j = 0;
        while (j < 512) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

// ============================================================
// 1024×1024 Matrix - Tile Size Sweep
// ============================================================

fn matmul_1024_tile32() -> i32 {
    i8<1024, 1024> A;
    i8<1024, 1024> B;
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            A[i as i64, j as i64] = ((i * 1024 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 1024 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(32, 32, 32) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_1024_tile64() -> i32 {
    i8<1024, 1024> A;
    i8<1024, 1024> B;
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            A[i as i64, j as i64] = ((i * 1024 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 1024 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(64, 64, 64) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_1024_tile128() -> i32 {
    i8<1024, 1024> A;
    i8<1024, 1024> B;
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            A[i as i64, j as i64] = ((i * 1024 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 1024 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(128, 128, 128) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn matmul_1024_tile256() -> i32 {
    i8<1024, 1024> A;
    i8<1024, 1024> B;
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            A[i as i64, j as i64] = ((i * 1024 + j) % 127 - 64) as i8;
            B[j as i64, i as i64] = ((i * 1024 + j) % 127 - 64) as i8;
            j = j + 1;
        }
        i = i + 1;
    }
    @tile(256, 256, 256) @lower("vnni.i8_matmul")
    var C = tensor_matmul(A, B);
    var sum = 0;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            sum = sum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }
    return sum;
}

fn kernel_main() -> i32 {
    return matmul_256_tile32();
}
