// Function arguments and calls test for MLIR lowering
// Tests: function args, variable lookups, function calls

fn add(var a, var b) {
    return a + b;
}

fn multiply(var x, var y) {
    return x * y;
}

fn compute(var a, var b, var c) {
    var sum = a + b;
    var product = sum * c;
    return product;
}
