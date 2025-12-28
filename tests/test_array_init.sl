// Test different array initialization approaches

fn main() -> f32 {
    // Method 1: Create and manually set all elements
    var A = array<f32>([4]);
    A[0] = 1.0;
    A[1] = 2.0;
    A[2] = 3.0;
    A[3] = 4.0;

    // Verify all elements are set
    var sum = A[0] + A[1] + A[2] + A[3];

    return sum;  // Should be 10.0
}
