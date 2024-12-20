fn bounded_sum(var n) {
    var sum = 0.0;
    var i = 1.0;
    
    while (i <= n) {
        sum = (sum + i) % 10000.0;
        i = i + 1.0;
    }
    return sum;
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

fn kernel_main() {
    var n = 100000.0;
    var sum_result = bounded_sum(n);
    var mul_result = bounded_multiply(n);
    return sum_result + mul_result;
} 