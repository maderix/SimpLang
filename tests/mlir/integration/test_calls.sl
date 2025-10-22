// Function calls test for MLIR lowering
// Tests: calling other functions, passing arguments

fn add(var a, var b) {
    return a + b;
}

fn multiply(var x, var y) {
    return x * y;
}

fn main() {
    var x = 10.0;
    var y = 5.0;
    var sum = add(x, y);
    var product = multiply(sum, y);
    return product;
}
