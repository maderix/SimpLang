// Test loop accumulation patterns with proper iter_args

// Sum of squares
fn sum_of_squares(var n) {
    var i = 1.0;
    var sum = 0.0;

    while (i <= n) {
        var square = i * i;
        sum = sum + square;
        i = i + 1.0;
    }

    return sum;
}

// Fibonacci number using while loop
fn fibonacci(var n) {
    var prev = 0.0;
    var curr = 1.0;
    var count = 2.0;
    var result = n;

    // Avoid early return - use result variable instead
    if (n > 1.0) {
        while (count <= n) {
            var next = prev + curr;
            prev = curr;
            curr = next;
            count = count + 1.0;
        }
        result = curr;
    }

    return result;
}

// Multiple accumulators in one loop
fn multiple_accumulators(var n) {
    var i = 1.0;
    var sum = 0.0;
    var product = 1.0;
    var count = 0.0;

    while (i <= n) {
        sum = sum + i;
        product = product * i;
        count = count + 1.0;
        i = i + 1.0;
    }

    // Return sum + product/count as a combined result
    return sum + product / count;
}

// Test loop with conditional accumulation
fn conditional_sum(var n) {
    var i = 0.0;
    var even_sum = 0.0;
    var odd_sum = 0.0;

    while (i < n) {
        // Check if even (simple approximation since we don't have modulo)
        var half = i / 2.0;
        var doubled = half * 2.0;

        if (i == doubled) {
            even_sum = even_sum + i;
        } else {
            odd_sum = odd_sum + i;
        }

        i = i + 1.0;
    }

    return even_sum - odd_sum;
}