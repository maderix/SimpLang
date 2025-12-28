// Simple INT8 matmul test case for VNNI debugging
// Uses tensor matmul which generates the 3-loop pattern

fn kernel_main() {
    // Small matrices for debugging: 8x32 * 32x8 = 8x8
    i8<8, 32> A;
    i8<32, 8> B;

    // Initialize A: each row is filled with (row_index + 1)
    var i = 0;
    while (i < 8) {
        var k = 0;
        while (k < 32) {
            A[i as i64, k as i64] = (i + 1) as i8;  // Row 0 = 1, Row 1 = 2, etc.
            k = k + 1;
        }
        i = i + 1;
    }

    // Initialize B: each column is filled with (col_index + 1)
    var k = 0;
    while (k < 32) {
        var j = 0;
        while (j < 8) {
            B[k as i64, j as i64] = (j + 1) as i8;  // Col 0 = 1, Col 1 = 2, etc.
            j = j + 1;
        }
        k = k + 1;
    }

    // Matrix multiply: C = A * B
    // This generates the pattern: for i, j: sum over k of A[i,k] * B[k,j]
    var C = tensor_matmul(A, B);

    // Expected: C[i,j] = (i+1) * (j+1) * 32
    // C[0,0] = 1*1*32 = 32
    // C[0,1] = 1*2*32 = 64
    // C[1,1] = 2*2*32 = 128

    // Sum all elements for verification
    var checksum = 0;
    i = 0;
    while (i < 8) {
        var j = 0;
        while (j < 8) {
            checksum = checksum + C[i as i64, j as i64];
            j = j + 1;
        }
        i = i + 1;
    }

    // Expected checksum:
    // Sum of (i+1)*(j+1)*32 for i,j in 0..7
    // = 32 * sum((i+1)*(j+1)) = 32 * (sum(i+1))^2 = 32 * 36^2 = 32 * 1296 = 41472
    print(checksum);
}
