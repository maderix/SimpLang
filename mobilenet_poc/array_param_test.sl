// Minimal array parameter test

fn test_array_param(f32[] arr, i32 count) -> f32 {
    if (count > 0) {
        return arr[0];
    }
    return 0.0;
}

fn kernel_main() -> f32 {
    return 42.0;
}