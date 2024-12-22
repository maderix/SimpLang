fn factorial(var n) {
    if (n <= 1.0) {
        return 1.0;
    }
    return n * factorial(n - 1.0);
}

fn fibonacci(var n) {
    if (n <= 1.0) {
        return n;
    }
    return fibonacci(n - 1.0) + fibonacci(n - 2.0);
}

fn sum_to_n(var n) {
    var sum = 0.0;
    var i = 1.0;
    while (i <= n) {
        sum = sum + i;
        i = i + 1.0;
    }
    return sum;
}

fn iterative_power(var base, var exp) {
    var result = 1.0;
    var i = 0.0;
    while (i < exp) {
        result = result * base;
        i = i + 1.0;
    }
    return result;
}

fn bounded_multiply(var n) {
    var result = 1.0;
    var i = 1.0;
    while (i <= n) {
        result = (result * i) % 10000.0;
        i = i + 1.0;
    }
    return result;
}

fn matrix_multiply(var size) {
    // Initialize matrices with size x size elements
    var A = 0.0;  // We'll use a flat array and calculate indices
    var B = 0.0;
    var C = 0.0;
    var sum = 0.0;
    var i = 0.0;
    var j = 0.0;
    var k = 0.0;
    
    // Matrix multiplication
    while (i < size) {
        j = 0.0;
        while (j < size) {
            k = 0.0;
            while (k < size) {
                // C[i][j] += A[i][k] * B[k][j]
                C = C + ((i + k) * (k * j));  // Simplified calculation
                k = k + 1.0;
            }
            sum = sum + C;  // Accumulate for verification
            j = j + 1.0;
        }
        i = i + 1.0;
    }
    return sum;
}

fn newton_sqrt(var x, var iterations) {
    var guess = x / 2.0;
    var i = 0.0;
    
    while (i < iterations) {
        guess = (guess + x / guess) / 2.0;
        i = i + 1.0;
    }
    return guess;
}

fn prime_sieve(var n) {
    var count = 0.0;
    var i = 2.0;
    var j = 0.0;
    var is_prime = 0.0;
    
    while (i <= n) {
        is_prime = 1.0;
        j = 2.0;
        while (j * j <= i) {
            if ((i % j) == 0.0) {
                is_prime = 0.0;
                j = i;  // Break
            }
            j = j + 1.0;
        }
        if (is_prime > 0.0) {
            count = count + 1.0;
        }
        i = i + 1.0;
    }
    return count;
}

fn kernel_main() {
    // Test different computational patterns
    var fact20 = factorial(20.0);     // Recursive
    var fib15 = fibonacci(15.0);      // Recursive with multiple calls
    var sum100 = sum_to_n(100.0);     // Simple iteration
    var pow_test = iterative_power(1.001, 1000.0);  // Many floating-point operations
    var bounded = bounded_multiply(50.0);  // Modulo arithmetic
    var matrix = matrix_multiply(10.0);    // Matrix operations
    var primes = prime_sieve(1000.0);      // Array operations and math
    var sqrt_test = newton_sqrt(2.0, 20.0); // Iterative numerical method
    
    // Return combined result to verify correctness
    return fact20 + fib15 + sum100 + pow_test + bounded + matrix + primes + sqrt_test;
} 