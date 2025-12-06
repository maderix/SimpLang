// Attention × V MatMul Benchmark for LLaMA 3.2 1B Self-Attention
// Attn: [seq_len, seq_len] = [1024, 1024] (attention weights after softmax)
// V: [seq_len, head_dim] = [1024, 64]
// Output: [seq_len, head_dim] = [1024, 64]
//
// This computes: output = softmax(QK^T) @ V
// INT8 quantized: i8 × i8 → i32

// Single head Attn×V: [1024, 1024] × [1024, 64] → [1024, 64]
fn benchmark_attn_v_matmul_1head() -> i32 {
    i8<1024, 1024> Attn;
    i8<1024, 64> V;

    // Initialize with test pattern
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            var val = ((i * 1024 + j) % 127) - 64;
            Attn[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            var val = ((i * 64 + j) % 127) - 64;
            V[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    // Attn × V matmul
    var output = tensor_matmul(Attn, V);

    // Checksum
    var checksum = 0;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            checksum = checksum + output[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}

// 4 heads Attn×V
fn benchmark_attn_v_matmul_4head() -> i32 {
    var total_checksum = 0;

    // Head 0
    i8<1024, 1024> Attn0;
    i8<1024, 64> V0;
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            var idx = 0 * 1024 * 1024 + i * 1024 + j;
            var val = (idx % 127) - 64;
            Attn0[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            var idx = 0 * 1024 * 64 + i * 64 + j;
            var val = (idx % 127) - 64;
            V0[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    var out0 = tensor_matmul(Attn0, V0);
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            total_checksum = total_checksum + out0[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    // Head 1
    i8<1024, 1024> Attn1;
    i8<1024, 64> V1;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            var idx = 1 * 1024 * 1024 + i * 1024 + j;
            var val = (idx % 127) - 64;
            Attn1[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            var idx = 1 * 1024 * 64 + i * 64 + j;
            var val = (idx % 127) - 64;
            V1[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    var out1 = tensor_matmul(Attn1, V1);
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            total_checksum = total_checksum + out1[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    // Head 2
    i8<1024, 1024> Attn2;
    i8<1024, 64> V2;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            var idx = 2 * 1024 * 1024 + i * 1024 + j;
            var val = (idx % 127) - 64;
            Attn2[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            var idx = 2 * 1024 * 64 + i * 64 + j;
            var val = (idx % 127) - 64;
            V2[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    var out2 = tensor_matmul(Attn2, V2);
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            total_checksum = total_checksum + out2[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    // Head 3
    i8<1024, 1024> Attn3;
    i8<1024, 64> V3;
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 1024) {
            var idx = 3 * 1024 * 1024 + i * 1024 + j;
            var val = (idx % 127) - 64;
            Attn3[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            var idx = 3 * 1024 * 64 + i * 64 + j;
            var val = (idx % 127) - 64;
            V3[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }
    var out3 = tensor_matmul(Attn3, V3);
    i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            total_checksum = total_checksum + out3[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return total_checksum;
}
