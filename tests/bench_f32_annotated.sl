// F32 Matmul Benchmark with Annotations
// Comparing different optimization strategies for 1024x1024 matrices
//
// Annotations tested:
//   @tile(M, K, N)     - Cache-friendly tiling
//   @parallel          - Parallel outer loops
//   @prefetch(N)       - Memory prefetching
//   @unroll(N)         - Loop unrolling

// ============================================================================
// 1024x1024 F32 Matmul Variants
// ============================================================================

// Baseline: No annotations (uses default 16x16x16 tiling)
fn matmul_1024_baseline() -> f32 {
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

// Tile 32x32x32
fn matmul_1024_tile32() -> f32 {
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

    @tile(32, 32, 32)
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

// Tile 64x64x64
fn matmul_1024_tile64() -> f32 {
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

    @tile(64, 64, 64)
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

// Tile 128x128x128
fn matmul_1024_tile128() -> f32 {
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

    @tile(128, 128, 128)
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

// Tile 256x256x256
fn matmul_1024_tile256() -> f32 {
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

    @tile(256, 256, 256)
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

// Parallel + Tile 64x64x64
fn matmul_1024_parallel_tile64() -> f32 {
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

    @parallel @tile(64, 64, 64)
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

// Parallel + Tile 128x128x128
fn matmul_1024_parallel_tile128() -> f32 {
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

    @parallel @tile(128, 128, 128)
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

// Tile + Prefetch
fn matmul_1024_tile64_prefetch() -> f32 {
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

    @tile(64, 64, 64) @prefetch(4)
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

// Full optimization: Parallel + Tile + Prefetch + Unroll
fn matmul_1024_full_opt() -> f32 {
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

    @parallel @tile(64, 64, 64) @prefetch(4) @unroll(4)
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

// Vectorize hint (256-bit AVX)
fn matmul_1024_tile64_vec256() -> f32 {
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

    @tile(64, 64, 64) @vectorize(256)
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

// ============================================================================
// 512x512 F32 Matmul Variants (for comparison)
// ============================================================================

fn matmul_512_baseline() -> f32 {
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

fn matmul_512_tile64() -> f32 {
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

    @tile(64, 64, 64)
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

fn matmul_512_parallel_tile64() -> f32 {
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

    @parallel @tile(64, 64, 64)
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

fn kernel_main() -> f32 {
    return matmul_1024_baseline();
}
