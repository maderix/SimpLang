fn main() {
    var sum = 0.0;
    var i = 1.0;
    
    while (i <= 5.0) {
        sum = sum + i;
        i = i + 1.0;
    }
    
    return sum;  // Should return 15.0 (1+2+3+4+5)
}
