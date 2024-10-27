fn fibonacci(var n) {
    if (n <= 1.0) {
        return n;
    }
    return fibonacci(n - 1.0) + fibonacci(n - 2.0);
}

fn main() {
    var n = 10.0;
    return fibonacci(n);  // Should return 55.0
}
