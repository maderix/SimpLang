// Test while loop with integer comparison
fn main() -> f32 {
    var i = 0;
    var sum = 0.0;

    while (i < 10) {
        sum = sum + 1.0;
        i = i + 1;
    }

    return sum;  // Should be 10.0
}
