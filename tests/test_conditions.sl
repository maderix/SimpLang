fn max(var a, var b) {
    if (a > b) {
        return a;
    }
    return b;
}

fn test_simple_if(var x) {
    if (x > 10.0) {
        return x;
    }
    return 10.0;
}

fn kernel_main() {
    // Test basic max function
    var result1 = max(15.0, 10.0);  // Should be 15.0
    
    // Test simple if
    var result2 = test_simple_if(20.0);  // Should be 20.0
    
    // Return sum
    return result1 + result2;  // Should be 35.0
}