fn max(var a, var b, var c) {
    var result = a;
    if (b > result) {
        result = b;
    }
    if (c > result) {
        result = c;
    }
    return result;
}

fn main() {
    var x = 10.0;
    var y = 20.0;
    var z = 15.0;
    return max(x, y, z);  // Should return 20.0
}
