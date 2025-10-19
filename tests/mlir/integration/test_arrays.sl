// Array operations test for MLIR lowering
// Demonstrates: array creation, indexing, stores, loads
// Progressive lowering: simp.array_* → memref.* → llvm.*

fn test_array_basic() f64 {
    // Create array: simp.array_create
    var arr = array<f64>([10]);

    // Array stores: simp.array_set
    arr[0] = 1.0;
    arr[1] = 2.0;
    arr[2] = 3.0;

    // Array loads: simp.array_get
    var sum = arr[0] + arr[1] + arr[2];

    return sum;  // Expected: 6.0
}

fn test_array_compute() f64 {
    var data = array<f64>([5]);

    // Initialize array elements
    data[0] = 10.0;
    data[1] = 20.0;
    data[2] = 30.0;

    // Compute with array elements
    var result = data[0] * data[1];  // 10 * 20 = 200
    result = result + data[2];        // 200 + 30 = 230

    return result;  // Expected: 230.0
}

fn test_array_accumulate() f64 {
    var values = array<f64>([4]);

    values[0] = 5.0;
    values[1] = 10.0;
    values[2] = 15.0;
    values[3] = 20.0;

    // Accumulate: demonstrates load-compute-store pattern
    var acc = 0.0;
    acc = acc + values[0];
    acc = acc + values[1];
    acc = acc + values[2];
    acc = acc + values[3];

    return acc;  // Expected: 50.0
}
