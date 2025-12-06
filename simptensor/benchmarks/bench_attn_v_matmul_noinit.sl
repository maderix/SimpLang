// Attention × V MatMul Benchmark - NO INIT version
// Only measures matmul time, data passed from outside as flat arrays

// Single head Attn×V: [1024, 1024] × [1024, 64] → [1024, 64]
fn benchmark_attn_v_noinit_1head(i8[] Attn_arr, i8[] V_arr) -> i32 {
    // Convert arrays to tensors
    i8<1024, 1024> Attn = tensor_from_array(Attn_arr, 0i);
    i8<1024, 64> V = tensor_from_array(V_arr, 0i);

    // Just matmul - no init
    var output = tensor_matmul(Attn, V);

    // Checksum
    var checksum = 0;
    var i = 0;
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

// 4 heads version
fn benchmark_attn_v_noinit_4head(
    i8[] Attn0_arr, i8[] V0_arr,
    i8[] Attn1_arr, i8[] V1_arr,
    i8[] Attn2_arr, i8[] V2_arr,
    i8[] Attn3_arr, i8[] V3_arr
) -> i32 {
    var total_checksum = 0;

    // Head 0
    i8<1024, 1024> Attn0 = tensor_from_array(Attn0_arr, 0i);
    i8<1024, 64> V0 = tensor_from_array(V0_arr, 0i);
    var out0 = tensor_matmul(Attn0, V0);
    var i = 0;
    while (i < 1024) {
        var j = 0;
        while (j < 64) {
            total_checksum = total_checksum + out0[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    // Head 1
    i8<1024, 1024> Attn1 = tensor_from_array(Attn1_arr, 0i);
    i8<1024, 64> V1 = tensor_from_array(V1_arr, 0i);
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
    i8<1024, 1024> Attn2 = tensor_from_array(Attn2_arr, 0i);
    i8<1024, 64> V2 = tensor_from_array(V2_arr, 0i);
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
    i8<1024, 1024> Attn3 = tensor_from_array(Attn3_arr, 0i);
    i8<1024, 64> V3 = tensor_from_array(V3_arr, 0i);
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
