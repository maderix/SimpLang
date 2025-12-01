// ARM Cross-Compilation Test Program
// This test validates basic SimpleLang functionality on ARM architecture

fn kernel_main() {
    // Test basic arithmetic
    var x = 10.0;
    var y = 20.0;
    var result = x + y;

    // Test multiplication
    result = result * 2.0;

    // Test division
    result = result / 3.0;

    // Expected result: (10 + 20) * 2 / 3 = 20.0
    return result;
}

// Test function calls
fn add(var a float64, var b float64) float64 {
    return a + b;
}

fn multiply(var a float64, var b float64) float64 {
    return a * b;
}

// Test arrays
fn array_test() float64 {
    var arr = make(Float64Array, 5);
    arr[0i] = 1.0;
    arr[1i] = 2.0;
    arr[2i] = 3.0;
    arr[3i] = 4.0;
    arr[4i] = 5.0;

    var sum = 0.0;
    var i = 0i;
    while (i < 5i) {
        sum = sum + arr[i];
        i = i + 1i;
    }

    // Expected sum: 1 + 2 + 3 + 4 + 5 = 15.0
    return sum;
}

// Test loops
fn loop_test() float64 {
    var count = 0.0;
    var i = 0i;

    while (i < 100i) {
        count = count + 1.0;
        i = i + 1i;
    }

    // Expected: 100.0
    return count;
}

// Main test function
fn test_all() float64 {
    var test1 = kernel_main();
    var test2 = add(5.0, 7.0);
    var test3 = multiply(3.0, 4.0);
    var test4 = array_test();
    var test5 = loop_test();

    // Combine all results: 20 + 12 + 12 + 15 + 100 = 159.0
    return test1 + test2 + test3 + test4 + test5;
}