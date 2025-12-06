// INT8 MatMul Parallel - Multiple Sizes
// Each size has chunk function for 8-way parallelism

// === 256x256 (chunk=32 rows) ===
fn int8_chunk_256(i8[] A, i8[] B, i32[] C, i64 row_start) -> i32 {
    i8<32, 256> A_local = tensor_from_array(A, row_start * 256i);
    i8<256, 256> B_local = tensor_from_array(B, 0i);
    var R = tensor_matmul(A_local, B_local);
    var i = 0i;
    while (i < 32i) {
        var j = 0i;
        while (j < 256i) { C[(row_start + i) * 256i + j] = R[i, j]; j = j + 1i; }
        i = i + 1i;
    }
    return 0;
}

// === 512x512 (chunk=64 rows) ===
fn int8_chunk_512(i8[] A, i8[] B, i32[] C, i64 row_start) -> i32 {
    i8<64, 512> A_local = tensor_from_array(A, row_start * 512i);
    i8<512, 512> B_local = tensor_from_array(B, 0i);
    var R = tensor_matmul(A_local, B_local);
    var i = 0i;
    while (i < 64i) {
        var j = 0i;
        while (j < 512i) { C[(row_start + i) * 512i + j] = R[i, j]; j = j + 1i; }
        i = i + 1i;
    }
    return 0;
}

// === 768x768 (chunk=96 rows) ===
fn int8_chunk_768(i8[] A, i8[] B, i32[] C, i64 row_start) -> i32 {
    i8<96, 768> A_local = tensor_from_array(A, row_start * 768i);
    i8<768, 768> B_local = tensor_from_array(B, 0i);
    var R = tensor_matmul(A_local, B_local);
    var i = 0i;
    while (i < 96i) {
        var j = 0i;
        while (j < 768i) { C[(row_start + i) * 768i + j] = R[i, j]; j = j + 1i; }
        i = i + 1i;
    }
    return 0;
}

// === 1024x1024 (chunk=128 rows) ===
fn int8_chunk_1024(i8[] A, i8[] B, i32[] C, i64 row_start) -> i32 {
    i8<128, 1024> A_local = tensor_from_array(A, row_start * 1024i);
    i8<1024, 1024> B_local = tensor_from_array(B, 0i);
    var R = tensor_matmul(A_local, B_local);
    var i = 0i;
    while (i < 128i) {
        var j = 0i;
        while (j < 1024i) { C[(row_start + i) * 1024i + j] = R[i, j]; j = j + 1i; }
        i = i + 1i;
    }
    return 0;
}

// === 2048x2048 (chunk=256 rows) ===
fn int8_chunk_2048(i8[] A, i8[] B, i32[] C, i64 row_start) -> i32 {
    i8<256, 2048> A_local = tensor_from_array(A, row_start * 2048i);
    i8<2048, 2048> B_local = tensor_from_array(B, 0i);
    var R = tensor_matmul(A_local, B_local);
    var i = 0i;
    while (i < 256i) {
        var j = 0i;
        while (j < 2048i) { C[(row_start + i) * 2048i + j] = R[i, j]; j = j + 1i; }
        i = i + 1i;
    }
    return 0;
}

// Full single-threaded matmul (uses VNNI)
fn int8_matmul_1024_full() -> i32 {
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

fn kernel_main() -> i32 {
    return int8_matmul_1024_full();
}
