// QK^T MatMul Benchmark for LLaMA 3.2 1B Self-Attention
// Q: [seq_len, head_dim] = [1024, 64]
// K^T: [head_dim, seq_len] = [64, 1024]
// Output: [seq_len, seq_len] = [1024, 1024]
//
// This is the attention score computation: scores = Q @ K^T
// INT8 quantized: i8 * i8 -> i32

// Single head QK^T: [1024, 64] x [64, 1024] -> [1024, 1024]
fn benchmark_qk_matmul_1head() -> i32 {
    i8<1024, 64> Q;
    i8<64, 1024> K_T;

    // Initialize with test pattern
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            var val = ((i * 64 + j) % 127) - 64;
            Q[i as i64, j as i64] = val;
            K_T[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    // QK^T matmul
    var scores = tensor_matmul(Q, K_T);

    // Checksum
    var checksum = 0;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            checksum = checksum + scores[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

// 4 heads - run single head 4 times (batch simulation)
fn benchmark_qk_matmul_4head() -> i32 {
    var total_checksum = 0;

    // Head 0
    i8<1024, 64> Q0;
    i8<64, 1024> K_T0;
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            var idx = 0 * 1024 * 64 + i * 64 + j;
            var val = (idx % 127) - 64;
            Q0[i as i64, j as i64] = val;
            K_T0[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    var scores0 = tensor_matmul(Q0, K_T0);
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            total_checksum = total_checksum + scores0[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    // Head 1
    i8<1024, 64> Q1;
    i8<64, 1024> K_T1;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            var idx = 1 * 1024 * 64 + i * 64 + j;
            var val = (idx % 127) - 64;
            Q1[i as i64, j as i64] = val;
            K_T1[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    var scores1 = tensor_matmul(Q1, K_T1);
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            total_checksum = total_checksum + scores1[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    // Head 2
    i8<1024, 64> Q2;
    i8<64, 1024> K_T2;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            var idx = 2 * 1024 * 64 + i * 64 + j;
            var val = (idx % 127) - 64;
            Q2[i as i64, j as i64] = val;
            K_T2[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    var scores2 = tensor_matmul(Q2, K_T2);
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            total_checksum = total_checksum + scores2[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    // Head 3
    i8<1024, 64> Q3;
    i8<64, 1024> K_T3;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            var idx = 3 * 1024 * 64 + i * 64 + j;
            var val = (idx % 127) - 64;
            Q3[i as i64, j as i64] = val;
            K_T3[j as i64, i as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    var scores3 = tensor_matmul(Q3, K_T3);
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            total_checksum = total_checksum + scores3[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return total_checksum;
}
