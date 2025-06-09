fn test_conditional() {
    var x = 5.0;
    var y = 10.0;
    
    if (x < y) {
        return x + y;
    } else {
        return x - y;
    }
}

fn kernel_main() {
    var result = test_conditional();
    return result;
}