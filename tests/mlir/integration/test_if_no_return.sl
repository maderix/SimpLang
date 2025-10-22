// If test without early returns (MLIR limitation)
fn test_if_else(var x) {
    var result = 0.0;
    if (x < 5.0) {
        result = x * 2.0;
    } else {
        result = x / 2.0;
    }
    return result;
}
