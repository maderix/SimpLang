// Control flow test for MLIR lowering
// Tests: if statements, while loops

fn test_if(var x) {
    if (x < 10.0) {
        return x + 1.0;
    }
    return x;
}

fn test_if_else(var x) {
    if (x < 5.0) {
        return x * 2.0;
    } else {
        return x / 2.0;
    }
}

fn test_while(var n) {
    var i = 0.0;
    var sum = 0.0;
    while (i < n) {
        sum = sum + i;
        i = i + 1.0;
    }
    return sum;
}
