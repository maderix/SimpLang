include "lib1.sl";

fn kernel_main() -> f32 {
    var result1 = add_two(10.0, 5.0);       // Should be 15.0
    var result2 = multiply_by_three(4.0);    // Should be 12.0
    return result1 + result2;                // Should be 27.0
}
