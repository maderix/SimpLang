fn fibonacci_recursive(var n) {
    if (n <= 1.0) {
        return n;
    }
    return fibonacci_recursive(n - 1.0) + fibonacci_recursive(n - 2.0);
}

fn fibonacci_iterative(var n) {
    if (n <= 1.0) {
        return n;
    }
    
    var prev = 0.0;
    var curr = 1.0;
    var i = 2.0;
    
    while (i <= n) {
        var next = prev + curr;
        prev = curr;
        curr = next;
        i = i + 1.0;
    }
    
    return curr;
}

fn kernel_main() {
    // Test both recursive and iterative implementations
    // Testing with n = 10 should give us 55
    
    // Test 1: Recursive Fibonacci
    var recursive_result = fibonacci_recursive(10.0);  // Should be 55.0
    
    // Test 2: Iterative Fibonacci
    var iterative_result = fibonacci_iterative(10.0);  // Should also be 55.0
    
    // Return sum of both results (should be 110.0 if both are correct)
    return recursive_result + iterative_result;
}