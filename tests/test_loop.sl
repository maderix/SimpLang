fn sum_to_n(var n) {
    var sum = 0.0;
    var i = 1.0;
    
    while (i <= n) {
        sum = sum + i;
        i = i + 1.0;
    }
    return sum;
}

fn factorial(var n) {
    var result = 1.0;
    var i = n;
    
    while (i > 1.0) {
        result = result * i;
        i = i - 1.0;
    }
    return result;
}

fn kernel_main() {
    // Test 1: Sum numbers from 1 to 5
    var sum_result = sum_to_n(5.0);  // Should be 15.0 (1+2+3+4+5)
    
    // Test 2: Factorial of 5
    var fact_result = factorial(5.0);  // Should be 120.0 (5*4*3*2*1)
    
    // Return sum of results
    return sum_result + fact_result;  // Should be 135.0
}