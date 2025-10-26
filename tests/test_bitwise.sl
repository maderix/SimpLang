// Test bitwise operators (integer only)
fn kernel_main() {
    // Test AND
    var a = 12i;  // 1100 in binary
    var b = 10i;  // 1010 in binary
    var and_result = a & b;  // Should be 8 (1000)

    // Test OR
    var or_result = a | b;  // Should be 14 (1110)

    // Test XOR
    var xor_result = a ^ b;  // Should be 6 (0110)

    // Test left shift
    var lshift_result = 3i << 2i;  // Should be 12 (3 * 4)

    // Test right shift
    var rshift_result = 12i >> 2i;  // Should be 3 (12 / 4)

    // Test modulo
    var mod_result = 17i % 5i;  // Should be 2

    // Combine all results
    var total = and_result + or_result + xor_result + lshift_result + rshift_result + mod_result;
    // Expected: 8 + 14 + 6 + 12 + 3 + 2 = 45

    // Convert to float by adding 0.0 (triggers implicit conversion)
    var result = total + 0.0;
    return result;
}
