// Small Attn×V test: [256, 256] × [256, 64] → [256, 64]
fn benchmark_attn_v_small() -> i32 {
    i8<256, 256> Attn;
    i8<256, 64> V;

    var i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 256) {
            var val = ((i * 256 + j) % 127) - 64;
            Attn[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 64) {
            var val = ((i * 64 + j) % 127) - 64;
            V[i as i64, j as i64] = val;
            j = j + 1;
        }
        i = i + 1;
    }

    var output = tensor_matvecmul(Attn, V);

    var checksum = 0;
    i = 0;
    while (i < 256) {
        var j = 0;
        while (j < 64) {
            checksum = checksum + output[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    return checksum;
}
