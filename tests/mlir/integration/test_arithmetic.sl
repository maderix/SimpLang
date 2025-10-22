// Simple arithmetic test for MLIR lowering
// Tests: literals, binary ops, variables, return

fn test_add() {
    var a = 10.0;
    var b = 20.0;
    var result = a + b;
    return result;
}

fn test_multiply() {
    var x = 5.0;
    var y = 6.0;
    return x * y;
}

fn test_combined() {
    var a = 10.0;
    var b = 5.0;
    var sum = a + b;      // 15.0
    var prod = a * b;     // 50.0;
    var result = sum + prod;  // 65.0
    return result;
}
