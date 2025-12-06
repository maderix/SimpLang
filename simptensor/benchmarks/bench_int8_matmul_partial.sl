// INT8 Partial MatMul for OpenMP parallelization
// Each function processes a chunk of rows, can be called in parallel from C++

// Process rows [row_start, row_start + chunk_size) of A × B → C
// A[M,K], B[K,N], C[M,N] passed as flat arrays with offsets
fn int8_matmul_chunk_1024(
    i8[] A,        // Input matrix A [M, K]
    i8[] B,        // Input matrix B [K, N]
    i32[] C,       // Output matrix C [M, N]
    i64 row_start, // Starting row index
    i64 chunk_size,// Number of rows to process
    i64 M,         // Total rows in A
    i64 K,         // Inner dimension
    i64 N          // Columns in B/C
) -> i32 {
    // Process chunk_size rows starting at row_start
    var i = row_start;
    var row_end = row_start + chunk_size;

    while (i < row_end) {
        var j = 0i;
        while (j < N) {
            var sum = 0;
            var k = 0i;
            while (k < K) {
                // A[i,k] * B[k,j]
                var a_val = A[i * K + k];
                var b_val = B[k * N + j];
                sum = sum + (a_val as i32) * (b_val as i32);
                k = k + 1i;
            }
            C[i * N + j] = sum;
            j = j + 1i;
        }
        i = i + 1i;
    }

    return 0;
}

// Tensor-based partial matmul using tensor_slice
// More elegant but needs compiler support for dynamic slicing
fn int8_matmul_tensor_chunk(
    i8[] A_flat,
    i8[] B_flat,
    i32[] C_flat,
    i64 row_start,
    i64 chunk_size,
    i64 M,
    i64 K,
    i64 N
) -> i32 {
    // Create full tensors from arrays
    i8<1024, 1024> A = tensor_from_array(A_flat, 0i);
    i8<1024, 1024> B = tensor_from_array(B_flat, 0i);

    // Slice A to get rows [row_start:row_start+chunk_size, :]
    // Note: tensor_slice uses compile-time bounds, so we use loop instead
    var i = row_start;
    var row_end = row_start + chunk_size;

    while (i < row_end) {
        var j = 0i;
        while (j < N) {
            var sum = 0;
            var k = 0i;
            while (k < K) {
                var a_val = A[i, k];
                var b_val = B[k, j];
                sum = sum + (a_val as i32) * (b_val as i32);
                k = k + 1i;
            }
            C_flat[(i * N + j) as i64] = sum;
            j = j + 1i;
        }
        i = i + 1i;
    }

    return 0;
}

// Full matmul for single-threaded comparison (uses VNNI optimization)
fn int8_matmul_full_1024() -> i32 {
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
    return int8_matmul_full_1024();
}
