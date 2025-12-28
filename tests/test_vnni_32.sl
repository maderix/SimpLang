// Single 32x32 INT8 matmul for VNNI debugging

fn kernel_main() -> i32 {
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
